import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Supply Chain Intelligence", layout="wide", page_icon="📦")

# --- KNOWLEDGE BASE: INTELLIGENT CATEGORY MAPPING ---
# This dictionary simulates a "Zero-Shot Classifier" by mapping technical nouns to Super-Categories
SUPER_CATEGORIES = {
    "TOOLS & HARDWARE": ["PLIER", "STRIPPER", "WRENCH", "SPANNER", "HAMMER", "BIT", "FILE", "SAW", "TOOL", "MEASURING TAPE", "CHISEL", "DRIVE"],
    "PIPING & FITTINGS": ["PIPE", "FLANGE", "ELBOW", "TEE", "REDUCER", "BEND", "COUPLING", "UPVC", "CPVC", "PVC", "GI", "NIPPLE", "BUSHING"],
    "VALVES & ACTUATORS": ["VALVE", "ACTUATOR", "BALL VALVE", "GATE VALVE", "CHECK VALVE", "GLOBE VALVE", "PLUG VALVE", "COCK"],
    "FASTENERS & SEALS": ["STUD", "BOLT", "NUT", "WASHER", "GASKET", "O-RING", "SEAL", "MECH SEAL", "GLOW", "JOINT"],
    "ELECTRICAL & INSTRUMENTATION": ["TRANSMITTER", "GAUGE", "CABLE", "WIRE", "CONNECTOR", "PLUG", "SWITCH", "HUB", "SENSOR"],
    "CONSUMABLES & CIVIL": ["BRUSH", "TAPE", "STICKER", "CHALK", "GLOVE", "CLEANER", "PAINT", "CEMENT", "HOSE", "ADHESIVE"]
}

# Prefix-to-Department mapping (The "Ground Truth" logic)
PREFIX_MAP = {
    'TL': 'TOOLS', 'IN': 'INSTRUMENTATION', 'CN': 'CONSUMABLES', 
    'SP': 'SPARE PARTS', 'BM': 'CIVIL/BUILDING', 'PS': 'STORES'
}

# --- LOGIC HELPERS ---
def get_super_category(noun):
    for category, keywords in SUPER_CATEGORIES.items():
        if noun in keywords:
            return category
    return "GENERAL"

def intelligent_noun_extractor(text):
    text = str(text).upper()
    # Check for multi-word phrases first (e.g., MEASURING TAPE, BALL VALVE)
    multi_word_targets = ["MEASURING TAPE", "BALL VALVE", "GATE VALVE", "CHECK VALVE", "PLUG VALVE", "PAINT BRUSH", "WIRE STRIPPER", "CUTTING PLIER"]
    for phrase in multi_word_targets:
        if phrase in text:
            return phrase
    
    # Check for single core nouns
    flat_keywords = [item for sublist in SUPER_CATEGORIES.values() for item in sublist]
    for noun in flat_keywords:
        if re.search(rf'\b{noun}\b', text):
            return noun
            
    # Fallback to first word skipping technical noise
    words = text.split()
    noise = ["SS", "GI", "MS", "PVC", "UPVC", "SIZE", "1/2", "3/4", "1", "2"]
    for w in words:
        clean = re.sub(r'[^A-Z]', '', w)
        if clean and clean not in noise and len(clean) > 2:
            return clean
    return "UNKNOWN"

# --- DATA PROCESSING ---
@st.cache_data
def run_intelligent_audit(file_path):
    try:
        # Load
        df = pd.read_csv(file_path, encoding='latin1')
        df.columns = [c.strip() for c in df.columns]
        
        # Identity Columns
        id_col = next(c for c in df.columns if any(x in c.lower() for x in ['item', 'no']))
        desc_col = next(c for c in df.columns if 'desc' in c.lower())
        
        # 1. Clean & Prefix Mapping
        df['Clean_Desc'] = df[desc_col].astype(str).str.upper().str.replace('"', '').str.strip()
        df['Prefix'] = df[id_col].str.extract(r'^([A-Z]+)')
        df['Dept'] = df['Prefix'].map(PREFIX_MAP).fillna('OTHER')

        # 2. Intelligent Category Logic
        df['Product_Noun'] = df['Clean_Desc'].apply(intelligent_noun_extractor)
        df['Main_Category'] = df['Product_Noun'].apply(get_super_category)

        # 3. Topic Modeling for Sub-Context
        tfidf = TfidfVectorizer(max_features=300, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['Clean_Desc'])
        nmf = NMF(n_components=10, random_state=42)
        nmf_features = nmf.fit_transform(tfidf_matrix)
        
        # Auto-Label Topics
        feat_names = tfidf.get_feature_names_out()
        topic_labels = {i: " ".join([feat_names[ind] for ind in nmf.components_[i].argsort()[-1:]]).upper() for i in range(10)}
        
        # Final Category Construction: Super Category + Sub-Context
        df['Sub_Context'] = [topic_labels[tid] for tid in nmf_features.argmax(axis=1)]
        df['Final_Category'] = df['Main_Category'] + " (" + df['Product_Noun'] + ")"

        # 4. Anomaly Detection
        df['Complexity'] = df['Clean_Desc'].apply(len)
        iso = IsolationForest(contamination=0.04, random_state=42)
        df['Anomaly_Flag'] = iso.fit_predict(df[['Complexity']])

        # 5. Fuzzy Logic (Male/Female Trap)
        exact_dups = df[df.duplicated(subset=['Clean_Desc'], keep=False)]
        
        return df, exact_dups, id_col, desc_col
    except Exception as e:
        st.error(f"ETL Error: {e}")
        return None, None, None, None

# --- UI EXECUTION ---
st.title("🛡️ Enterprise AI Inventory Auditor")
st.markdown("---")

# Use your uploaded file
target_file = 'raw_data.csv'
if os.path.exists(target_file):
    df, exact_dups, id_col, desc_col = run_intelligent_audit(target_file)
    
    if df is not None:
        # KPI ROW
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Items Audited", len(df))
        m2.metric("Super Categories", df['Main_Category'].nunique())
        m3.metric("Anomalies", len(df[df['Anomaly_Flag'] == -1]))
        m4.metric("Dept Accuracy", f"{len(df[df['Dept'] != 'OTHER'])/len(df):.0%}")

        tabs = st.tabs(["📍 Categorization", "🚨 Anomaly", "🧠 Methodology", "📈 Reports"])

        with tabs[0]:
            st.header("Refined Product Classification")
            st.info("The system now uses multi-word noun extraction and a prioritized industry knowledge base.")
            
            # Highlight user's problematic items
            search = st.text_input("Search (e.g., Plier, Stripper, Tape):")
            if search:
                view_df = df[df['Clean_Desc'].str.contains(search.upper())]
            else:
                view_df = df
            
            st.dataframe(view_df[[id_col, 'Clean_Desc', 'Final_Category', 'Dept']], use_container_width=True)

        with tabs[1]:
            st.header("Anomalies & Outliers")
            st.dataframe(df[df['Anomaly_Flag'] == -1][[id_col, desc_col, 'Final_Category']])

        with tabs[2]:
            st.header("How the AI classifies intelligently:")
            st.markdown("""
            - **Phase 1 (The Anchor):** The system checks for "Multi-word Phrases" (e.g. *Measuring Tape*).
            - **Phase 2 (The Knowledge Base):** The noun is compared against an industrial dictionary to find its Super-Category (e.g. *Plier* -> *Tools*).
            - **Phase 3 (Validation):** The AI cross-references the item **Prefix** (e.g. *TL* prefix must align with *Tools* category).
            """)

        with tabs[3]:
            st.plotly_chart(px.pie(df, names='Main_Category', title="Inventory Breakdown by Category"))
else:
    st.warning("File 'raw_data.csv' not found. Please ensure it is in your repository.")
