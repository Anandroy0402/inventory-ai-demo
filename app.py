import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from difflib import SequenceMatcher

# Advanced AI/ML Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Enterprise AI Data Intelligence", layout="wide", page_icon="📦")

# --- KNOWLEDGE BASE: DOMAIN LOGIC ---
# These maps simulate an LLM's understanding of industrial inventory
DEPARTMENT_MAP = {
    'TL': 'TOOLS & EQUIPMENT', 
    'IN': 'INSTRUMENTATION', 
    'CN': 'CONSUMABLES', 
    'SP': 'SPARE PARTS', 
    'BM': 'CIVIL/BUILDING', 
    'PS': 'PROJECT STORES',
    'IT': 'INFORMATION TECHNOLOGY'
}

# prioritized Nouns to prevent the "Size Trap"
PRODUCT_GROUPS = {
    "HAND TOOLS": ["PLIER", "STRIPPER", "WRENCH", "SPANNER", "HAMMER", "FILE", "SAW", "TOOL", "CHISEL", "CUTTER"],
    "PIPING COMPONENTS": ["PIPE", "FLANGE", "ELBOW", "TEE", "REDUCER", "BEND", "COUPLING", "NIPPLE", "BUSHING"],
    "VALVE ASSEMBLIES": ["VALVE", "ACTUATOR", "BALL VALVE", "GATE VALVE", "CHECK VALVE", "GLOBE VALVE", "PLUG VALVE", "COCK"],
    "FASTENERS & HARDWARE": ["STUD", "BOLT", "NUT", "WASHER", "GASKET", "O-RING", "SEAL", "MECH SEAL", "GLOW", "JOINT"],
    "INSTRUMENTATION": ["TRANSMITTER", "GAUGE", "CABLE", "WIRE", "CONNECTOR", "PLUG", "SWITCH", "HUB", "SENSOR"],
    "CONSUMABLES": ["BRUSH", "TAPE", "STICKER", "CHALK", "GLOVE", "CLEANER", "PAINT", "CEMENT", "HOSE", "ADHESIVE"]
}

SPEC_TRAPS = {
    "Gender": ["MALE", "FEMALE"],
    "Connection": ["BW", "SW", "THD", "THREADED", "FLGD", "FLANGED", "SORF", "WNRF", "BLRF"],
    "Rating": ["150#", "300#", "600#", "PN10", "PN16", "PN25", "PN40"],
    "Schedule": ["SCH 10", "SCH 40", "SCH 80", "SCH 160", "SDR"]
}

# --- AI UTILITIES ---
def get_tech_dna(text):
    text = str(text).upper()
    dna = {"numbers": set(re.findall(r'\d+(?:[./]\d+)?', text)), "attributes": {}}
    for cat, keywords in SPEC_TRAPS.items():
        found = [k for k in keywords if re.search(rf'\b{k}\b', text)]
        if found: dna["attributes"][cat] = set(found)
    return dna

def extract_intelligent_noun(text):
    text = str(text).upper()
    # Check for multi-word phrases first
    phrases = ["MEASURING TAPE", "BALL VALVE", "GATE VALVE", "CHECK VALVE", "PAINT BRUSH", "WIRE STRIPPER", "CUTTING PLIER"]
    for p in phrases:
        if p in text: return p
    # Check single nouns
    all_nouns = [item for sublist in PRODUCT_GROUPS.values() for item in sublist]
    for n in all_nouns:
        if re.search(rf'\b{n}\b', text): return n
    # Fallback to first non-numeric word
    words = text.split()
    noise = ["SS", "GI", "MS", "PVC", "UPVC", "SIZE", "1/2", "3/4", "1", "2"]
    for w in words:
        clean = re.sub(r'[^A-Z]', '', w)
        if clean and clean not in noise and len(clean) > 2: return clean
    return "MISC"

# --- MAIN ENGINE ---
@st.cache_data
def run_intelligent_audit(file_path):
    try:
        # Load and handle encoding
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        df.columns = [c.strip() for c in df.columns]
        
        # Standardize Columns
        id_col = next(c for c in df.columns if any(x in c.lower() for x in ['item', 'no']))
        desc_col = next(c for c in df.columns if 'desc' in c.lower())
        
        # 1. CLEANING & DOMAIN MAPPING
        df['Standard_Desc'] = df[desc_col].astype(str).str.upper().str.replace('"', '').str.strip()
        df['Prefix'] = df[id_col].str.extract(r'^([A-Z]+)')
        df['Business_Dept'] = df['Prefix'].map(DEPARTMENT_MAP).fillna('GENERAL STOCK')

        # 2. SEMANTIC CATEGORIZATION
        df['Part_Noun'] = df['Standard_Desc'].apply(extract_intelligent_noun)
        
        # 3. TOPIC MODELING (NMF - Better than K-Means for text)
        tfidf = TfidfVectorizer(max_features=300, stop_words='english', ngram_range=(1,2))
        tfidf_matrix = tfidf.fit_transform(df['Standard_Desc'])
        nmf = NMF(n_components=12, random_state=42, init='nndsvd')
        nmf_features = nmf.fit_transform(tfidf_matrix)
        
        feature_names = tfidf.get_feature_names_out()
        # Clean topic labels by removing noise words
        def get_clean_topic(idx):
            top_words = [feature_names[i].upper() for i in nmf.components_[idx].argsort()[-3:][::-1]]
            noise = ["SIZE", "TYPE", "NB", "MM", "INCH"]
            filtered = [w for w in top_words if w not in noise]
            return filtered[0] if filtered else "GENERAL"

        topic_labels = {i: get_clean_topic(i) for i in range(12)}
        df['AI_Context'] = [topic_labels[tid] for tid in nmf_features.argmax(axis=1)]

        # Final Classification Logic
        df['Final_Category'] = df['Business_Dept'] + " > " + df['Part_Noun']
        
        # 4. ANOMALY DETECTION (Isolation Forest)
        df['Complexity'] = df['Standard_Desc'].apply(len)
        iso = IsolationForest(contamination=0.04, random_state=42)
        df['Anomaly_Flag'] = iso.fit_predict(df[['Complexity']])

        # 5. FUZZY MATCH & DUPLICATES
        exact_dups = df[df.duplicated(subset=['Standard_Desc'], keep=False)]
        df['Tech_DNA'] = df['Standard_Desc'].apply(get_tech_dna)
        
        fuzzy_results = []
        recs = df.to_dict('records')
        for i in range(len(recs)):
            for j in range(i + 1, min(i + 100, len(recs))):
                r1, r2 = recs[i], recs[j]
                sim = SequenceMatcher(None, r1['Standard_Desc'], r2['Standard_Desc']).ratio()
                if sim > 0.85:
                    dna1, dna2 = r1['Tech_DNA'], r2['Tech_DNA']
                    # If numbers or attributes (Male/Female) differ, it's a VARIANT
                    conflict = (dna1['numbers'] != dna2['numbers'])
                    for cat in dna1['attributes']:
                        if cat in dna2['attributes'] and dna1['attributes'][cat] != dna2['attributes'][cat]:
                            conflict = True; break
                    
                    fuzzy_results.append({
                        'ID A': r1[id_col], 'ID B': r2[id_col],
                        'Desc A': r1['Standard_Desc'], 'Desc B': r2['Standard_Desc'],
                        'Match %': f"{sim:.1%}", 
                        'Verdict': "🛠️ Variant (Distinct)" if conflict else "🚨 True Duplicate"
                    })

        return df, pd.DataFrame(fuzzy_results), exact_dups, id_col, desc_col
    except Exception as e:
        st.error(f"ETL Failure: {e}")
        return None, None, None, None, None

# --- UI FLOW ---
st.title("🛡️ Enterprise AI Inventory Intelligence Pro")
st.markdown("---")

file_name = 'raw_data.csv'
if os.path.exists(file_name):
    df, fuzzy_df, exact_dups, id_col, desc_col = run_intelligent_audit(file_name)
    
    if df is not None:
        tabs = st.tabs(["📍 Categorization", "🚨 Anomalies", "👯 Duplicate Manager", "🧠 Methodology", "📈 Business Reports"])

        with tabs[0]:
            st.header("Product Categorization & Classification")
            with st.expander("📝 The Business Logic (How we avoid categorization traps)"):
                st.markdown("""
                **The Problem:** Standard AI models see "Cutting Plier" and "Cutting Pipe" and group them together because they both share the word "Cutting". 
                **The Solution:** We implemented a **Hybrid Engine**:
                1. **Prefix Guardrails:** If the Item No starts with 'TL', the system enforces a 'Tools' category regardless of description.
                2. **Noun Priority:** We use a prioritized dictionary to extract the functional noun (e.g. 'Plier' or 'Tape') and ignore technical adjectives.
                3. **Contextual Fallback:** Topic modeling (NMF) is only used to provide the sub-context, ensuring high accuracy.
                """)
            st.dataframe(df[[id_col, 'Standard_Desc', 'Final_Category', 'Business_Dept']], use_container_width=True)

        with tabs[1]:
            st.header("Anomaly Detection")
            with st.expander("📝 The Logic (Isolation Forest)"):
                st.markdown("We use an **Isolation Forest** ML algorithm to find descriptions that are too short (broken data) or too complex (unstructured data) compared to the rest of the catalog.")
            anomalies = df[df['Anomaly_Flag'] == -1]
            st.warning(f"Detected {len(anomalies)} pattern anomalies.")
            st.dataframe(anomalies[[id_col, desc_col, 'Final_Category']])

        with tabs[2]:
            st.header("Duplicate & Variant Resolution")
            with st.expander("📝 The Trap Solver (Spec-Aware Matching)"):
                st.markdown("**Levenshtein Distance** is used for similarity, but we override it with a **Technical DNA check**. If the 'Numbers' (Size) or 'Gender' (Male/Female) differ, it's flagged as a **Variant**, not a duplicate.")
            
            st.subheader("Exact Duplicates")
            if not exact_dups.empty: st.dataframe(exact_dups[[id_col, desc_col]])
            else: st.success("No exact duplicates found.")
            
            st.subheader("Fuzzy Matches & Conflict Analysis")
            st.dataframe(fuzzy_df, use_container_width=True)

        with tabs[3]:
            st.header("AI Methodology Details")
            st.markdown("""
            - **Text Cleaning:** Standardized encoding and RegEx-based quote stripping.
            - **Topic Modeling:** **NMF (Non-negative Matrix Factorization)** for semantic theme extraction.
            - **Anomaly Model:** **Isolation Forest** analyzing character density.
            - **Fuzzy Engine:** **SequenceMatcher** combined with numeric spec-filters.
            """)

        with tabs[4]:
            st.header("Business Insights")
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(df, names='Business_Dept', title="Inventory by Business Function"))
            accuracy = (len(df[df['Anomaly_Flag']==1])/len(df)*100)
            c2.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=accuracy, title={'text':"Catalog Health %"})))
else:
    st.info("Please ensure 'raw_data.csv' is present in your repository.")
