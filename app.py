import streamlit as st
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Inventory Manager", layout="wide", page_icon="📦")

# --- HEADER ---
st.title("📦 AI Inventory Intelligence Platform")
st.markdown("""
**System Status:** 🟢 Online  
**Pipeline:** Cloud-Native ETL & ML  
**Logic:** Strict Separation of *ID Validation* (Exact) and *Description Matching* (Fuzzy).
""")

# --- ETL PIPELINE ---
@st.cache_data
def load_and_process_data(file_path):
    # 1. Ingestion
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
    except Exception as e:
        return None, f"Error reading file: {e}"

    # 2. Smart Column Detection
    df.columns = [c.strip() for c in df.columns]
    
    # Identify ID Column (Look for 'Item', 'No', 'ID', 'Code')
    id_col = next((c for c in df.columns if any(x in c.lower() for x in ['item', 'no.', 'id', 'code'])), df.columns[0])
    
    # Identify Description Column (Look for 'Desc', 'Name' or longest text column)
    desc_col = next((c for c in df.columns if 'desc' in c.lower()), None)
    if not desc_col:
        # Fallback: Find column with longest average string length
        text_cols = df.select_dtypes(include=['object'])
        if not text_cols.empty:
            desc_col = text_cols.apply(lambda x: x.str.len().mean()).idxmax()
        else:
            desc_col = df.columns[1]

    # 3. Data Cleaning
    def clean_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'"+', '', text) # Remove quote artifacts
        text = re.sub(r'[^a-z0-9\s./-]', ' ', text) # Keep alphanumeric + basic punctuation
        return re.sub(r'\s+', ' ', text).strip()

    df['Clean_Desc'] = df[desc_col].apply(clean_text)

    # 4. AI Categorization (K-Means)
    try:
        tfidf = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['Clean_Desc'])
        
        kmeans = KMeans(n_clusters=6, random_state=42)
        df['Cluster_ID'] = kmeans.fit_predict(tfidf_matrix)
        
        # Auto-Labeling
        terms = tfidf.get_feature_names_out()
        cluster_names = {}
        for i in range(6):
            center = kmeans.cluster_centers_[i]
            top_terms = [terms[ind] for ind in center.argsort()[-3:]]
            cluster_names[i] = " / ".join(top_terms).upper()
        df['AI_Category'] = df['Cluster_ID'].map(cluster_names)
    except:
        df['AI_Category'] = "Uncategorized"

    # 5. Anomaly Detection (Isolation Forest)
    # Features: Description Length, Digit Count (Complexity)
    df['Desc_Len'] = df['Clean_Desc'].apply(len)
    df['Digit_Count'] = df['Clean_Desc'].apply(lambda x: len(re.findall(r'\d', x)))
    
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly_Score'] = iso.fit_predict(df[['Desc_Len', 'Digit_Count']])
    df['Is_Anomaly'] = df['Anomaly_Score'].apply(lambda x: 'High Risk' if x == -1 else 'Normal')

    return df, id_col, desc_col

# --- DUPLICATE LOGIC ---
def run_duplicate_check(df, id_col, desc_col):
    duplicates = []
    
    # Check 1: Exact ID Duplicates (Critical Data Error)
    exact_id_dups = df[df.duplicated(subset=[id_col], keep=False)]
    if not exact_id_dups.empty:
        for i, row in exact_id_dups.iterrows():
             duplicates.append({
                'Type': 'CRITICAL: Same ID',
                'Item A': row[id_col],
                'Item B': row[id_col],
                'Description A': row[desc_col],
                'Description B': 'SAME ID EXISTS TWICE',
                'Score': '100%'
            })

    # Check 2: Fuzzy Description Matches (Potential SKU Merge)
    # Optimization: Convert to list of dicts for speed
    records = df.to_dict('records')
    limit = min(len(records), 600) # Cloud resource cap
    
    for i in range(limit):
        for j in range(i + 1, limit):
            # Strict Rule: Only compare if IDs are DIFFERENT
            if records[i][id_col] == records[j][id_col]:
                continue
                
            # Heuristic Blocking: Skip if length difference is huge
            if abs(len(records[i]['Clean_Desc']) - len(records[j]['Clean_Desc'])) > 8:
                continue
            
            # Levenshtein Ratio
            ratio = SequenceMatcher(None, records[i]['Clean_Desc'], records[j]['Clean_Desc']).ratio()
            
            if ratio > 0.88: # Threshold (88% similarity)
                duplicates.append({
                    'Type': 'WARNING: Fuzzy Match',
                    'Item A': records[i][id_col],
                    'Item B': records[j][id_col],
                    'Description A': records[i][desc_col],
                    'Description B': records[j][desc_col],
                    'Score': f"{ratio:.1%}"
                })
                
    return pd.DataFrame(duplicates)

# --- EXECUTION ---
# Load Data directly from Repo
file_path = 'raw_data.csv' 
try:
    df, id_col, desc_col = load_and_process_data(file_path)
except:
    st.warning("Data file 'raw_data.csv' not found in repository. Please upload it.")
    st.stop()

if isinstance(df, str): # Error catch
    st.error(df)
    st.stop()

# Calculate Duplicates
dups_df = run_duplicate_check(df, id_col, desc_col)

# --- DASHBOARD UI ---

# Top Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total SKU Count", len(df))
m2.metric("Categories Identified", df['AI_Category'].nunique())
m3.metric("Anomalies Flagged", len(df[df['Is_Anomaly']=='High Risk']))
m4.metric("Duplicate Conflicts", len(dups_df))

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 AI Categorization", "🚨 Anomaly Detection", "👯 Duplicate Manager"])

with tab1:
    st.subheader("Automated Inventory Clustering")
    st.markdown("Items grouped by semantic similarity using **Unsupervised Learning (K-Means)**.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Bar Chart
        chart_data = df['AI_Category'].value_counts().reset_index()
        chart_data.columns = ['Category', 'Count']
        fig = px.bar(chart_data, x='Count', y='Category', orientation='h', color='Count', title="Category Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(df[[id_col, desc_col, 'AI_Category']].head(100), height=400)

with tab2:
    st.subheader("Outlier Detection (Isolation Forest)")
    st.markdown("These items deviate significantly from standard description patterns (Length/Complexity).")
    
    anomalies = df[df['Is_Anomaly']=='High Risk']
    st.dataframe(anomalies[[id_col, desc_col, 'Desc_Len', 'Digit_Count']], use_container_width=True)
    
    # Scatter Visual
    fig2 = px.scatter(df, x='Desc_Len', y='Digit_Count', color='Is_Anomaly', 
                      title="Anomaly Visualization: Description Length vs. Digit Count",
                      color_discrete_map={'Normal':'#00CC96', 'High Risk':'#EF553B'},
                      hover_data=[desc_col])
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Inventory Conflict Resolution")
    st.info(f"**Policy:** Exact ID matches are CRITICAL errors. Fuzzy description matches (different IDs) are WARNINGS.")
    
    if not dups_df.empty:
        # Color code the dataframe based on Type
        def highlight_type(val):
            color = 'red' if 'CRITICAL' in val else 'orange'
            return f'color: {color}; font-weight: bold'
            
        st.dataframe(dups_df.style.applymap(highlight_type, subset=['Type']), use_container_width=True)
    else:
        st.success("✅ Clean Data! No duplicates detected.")
