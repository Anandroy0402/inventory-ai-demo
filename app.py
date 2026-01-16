import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from difflib import SequenceMatcher

# Machine Learning & AI Stack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# Hugging Face Transformers
try:
    from transformers import pipeline
except ImportError:
    st.error("Missing libraries! Please add 'transformers' and 'torch' to your requirements.txt")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Inventory Auditor Pro", layout="wide", page_icon="🛡️")

# --- KNOWLEDGE BASE: DOMAIN LOGIC ---
DEFAULT_PRODUCT_GROUP = "Consumables & General"

PRODUCT_GROUPS = {
    "Piping & Fittings": ["FLANGE", "PIPE", "ELBOW", "TEE", "UNION", "REDUCER", "BEND", "COUPLING", "NIPPLE", "BUSHING", "UPVC", "CPVC", "PVC"],
    "Valves & Actuators": ["BALL VALVE", "GATE VALVE", "PLUG VALVE", "CHECK VALVE", "GLOBE VALVE", "CONTROL VALVE", "VALVE", "ACTUATOR", "COCK"],
    "Fasteners & Seals": ["STUD", "BOLT", "NUT", "WASHER", "GASKET", "O RING", "MECHANICAL SEAL", "SEAL", "JOINT"],
    "Electrical & Instruments": ["TRANSMITTER", "CABLE", "WIRE", "GAUGE", "SENSOR", "CONNECTOR", "SWITCH", "TERMINAL", "INSTRUMENT", "CAMERA"],
    "Tools & Hardware": ["PLIER", "CUTTING PLIER", "STRIPPER", "WIRE STRIPPER", "WRENCH", "SPANNER", "HAMMER", "FILE", "SAW", "TOOL", "CHISEL", "CUTTER", "TAPE MEASURE", "MEASURING TAPE", "BIT", "DRILL BIT"],
    "Consumables & General": ["BRUSH", "PAINT BRUSH", "TAPE", "ADHESIVE", "HOSE", "SAFETY GLOVE", "GLOVE", "CLEANER", "PAINT", "CEMENT", "STICKER", "CHALK"],
    "Specialized Spares": ["FILTER", "BEARING", "PUMP", "MOTOR", "CARTRIDGE", "IMPELLER", "SPARE"]
}

SPEC_TRAPS = {
    "Gender": ["MALE", "FEMALE"],
    "Connection": ["BW", "SW", "THD", "THREADED", "FLGD", "FLANGED", "SORF", "WNRF", "BLRF"],
    "Rating": ["150#", "300#", "600#", "PN10", "PN16", "PN25", "PN40"]
}

# --- AI MODELS LOADING ---
@st.cache_resource
def load_hf_classifier():
    # Using a CPU-optimized, lightweight model for Zero-Shot Classification
    return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

# --- AI UTILITIES ---
def clean_description(text):
    text = str(text).upper().replace('"', ' ')
    text = text.replace("O-RING", "O RING")
    text = text.replace("MECH-SEAL", "MECHANICAL SEAL").replace("MECH SEAL", "MECHANICAL SEAL")
    text = re.sub(r'[^A-Z0-9\s./-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def token_pattern(token):
    return rf'(?<!\w){re.escape(token)}(?!\w)'

def get_tech_dna(text):
    text = clean_description(text)
    dna = {"numbers": set(re.findall(r'\d+(?:[./]\d+)?', text)), "attributes": {}}
    for cat, keywords in SPEC_TRAPS.items():
        found = [k for k in keywords if re.search(token_pattern(k), text)]
        if found: dna["attributes"][cat] = set(found)
    return dna

# --- MAIN ENGINE ---
@st.cache_data
def run_intelligent_audit(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    df.columns = [c.strip() for c in df.columns]
    id_col = next(c for c in df.columns if any(x in c.lower() for x in ['item', 'no']))
    desc_col = next(c for c in df.columns if 'desc' in c.lower())
    
    df['Standard_Desc'] = df[desc_col].apply(clean_description)
    
    # 1. HUGGING FACE CATEGORIZATION
    classifier = load_hf_classifier()
    candidate_labels = list(PRODUCT_GROUPS.keys())
    
    # Processing only a sample or specific batch if needed for speed
    results = classifier(df['Standard_Desc'].tolist(), candidate_labels=candidate_labels)
    df['Product_Group'] = [r['labels'][0] for r in results]
    df['AI_Confidence'] = [round(r['scores'][0], 4) for r in results]

    # 2. VECTORIZATION & CLUSTERING
    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Standard_Desc'])
    
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(tfidf_matrix)
    dists = kmeans.transform(tfidf_matrix)
    df['Distance_Score'] = np.min(dists, axis=1)

    # 3. ANOMALY DETECTION
    iso = IsolationForest(contamination=0.04, random_state=42)
    df['Anomaly_Flag'] = iso.fit_predict(tfidf_matrix)

    # 4. TECH DNA
    df['Tech_DNA'] = df['Standard_Desc'].apply(get_tech_dna)
    
    return df, id_col, desc_col

# --- DATA LOADING ---
target_file = 'raw_data.csv'
if os.path.exists(target_file):
    df_raw, id_col, desc_col = run_intelligent_audit(target_file)
else:
    st.error("raw_data.csv not found!")
    st.stop()

# --- TABS ---
st.title("🛡️ AI Inventory Auditor Pro")
tabs = st.tabs(["📈 Executive Dashboard", "📍 AI Categorization", "🚨 Quality Hub", "🧠 Methodology"])

with tabs[0]:
    st.header("Inventory Data Health")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("SKUs Analyzed", len(df_raw))
    kpi2.metric("Avg AI Confidence", f"{df_raw['AI_Confidence'].mean():.1%}")
    kpi3.metric("Pattern Anomalies", len(df_raw[df_raw['Anomaly_Flag'] == -1]))

    # Distribution Chart
    fig_pie = px.pie(df_raw, names='Product_Group', title="Categorization Distribution", hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

with tabs[1]:
    st.header("Semantic Classification Details")
    with st.expander("Why Hugging Face?"):
        st.write("Traditional keyword matching fails when a 'Stripper' is mentioned without the word 'Tool'. Hugging Face understands the **meaning** (semantics) of the words to bucket them correctly.")
    st.dataframe(df_raw[[id_col, 'Standard_Desc', 'Product_Group', 'AI_Confidence']])

with tabs[2]:
    st.header("Quality & Risk Identification")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Anomalies")
        anoms = df_raw[df_raw['Anomaly_Flag'] == -1]
        st.dataframe(anoms[[id_col, 'Standard_Desc']])
    
    with col2:
        st.subheader("👯 Fuzzy Match (Spec-Aware)")
        # Simple fuzzy duplicate logic
        fuzzy_list = []
        recs = df_raw.head(100).to_dict('records') # Optimized for demo speed
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                sim = SequenceMatcher(None, recs[i]['Standard_Desc'], recs[j]['Standard_Desc']).ratio()
                if sim > 0.85:
                    dna1, dna2 = recs[i]['Tech_DNA'], recs[j]['Tech_DNA']
                    conflict = dna1['numbers'] != dna2['numbers']
                    fuzzy_list.append({'A': recs[i][id_col], 'B': recs[j][id_col], 'Match': f"{sim:.1%}", 'Status': "Variant" if conflict else "Duplicate"})
        st.dataframe(pd.DataFrame(fuzzy_list))

with tabs[3]:
    st.header("The AI Engine Breakdown")
    st.markdown("""
    - **Zero-Shot Classification**: DistilBERT model trained on MNLI dataset to infer category relationships without specific training on your data.
    - **TF-IDF & K-Means**: Numerical vectorization and clustering to identify semantic neighborhoods.
    - **Isolation Forest**: Analyzes the structure of descriptions to isolate 'outlier' data entries.
    """)
