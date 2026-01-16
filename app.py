import streamlit as st
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Inventory Auditor", layout="wide", page_icon="⚙️")

st.title("⚙️ AI Inventory Intelligence Platform")
st.markdown("""
**Data Integrity Policy:** This system uses **Numeric Fingerprinting**. If numeric values in a description differ (e.g., 1" vs 3"), the items are classified as **Variants**, protecting distinct SKUs from being merged.
""")

# --- LOGIC ---
def get_numeric_fingerprint(text):
    """Extracts all numbers (integers, decimals, fractions) as a signature."""
    if not isinstance(text, str): return set()
    # Captures 1, 1/2, 0.5, 10, etc.
    return set(re.findall(r'\d+(?:[./]\d+)?', text))

@st.cache_data
def run_audit(file_path):
    # 1. Load & Clean
    df = pd.read_csv(file_path)
    df.columns = ['Index', 'Item_No', 'Description', 'UoM']
    
    def deep_clean(text):
        text = str(text).upper()
        text = re.sub(r'"+', '', text) # Remove quote artifacts
        return " ".join(text.split()).strip()

    df['Clean_Desc'] = df['Description'].apply(deep_clean)
    # Generate the Numeric Fingerprint for every row
    df['Num_Fingerprint'] = df['Clean_Desc'].apply(get_numeric_fingerprint)

    # 2. AI Categorization
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Clean_Desc'])
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(tfidf_matrix)
    
    # Auto-label based on top keywords
    terms = tfidf.get_feature_names_out()
    labels = {i: " ".join([terms[ind] for ind in kmeans.cluster_centers_[i].argsort()[-2:]]).upper() for i in range(8)}
    df['Category'] = df['Cluster_ID'].map(labels)

    # 3. Anomaly Detection (Isolation Forest)
    df['Length'] = df['Clean_Desc'].apply(len)
    iso = IsolationForest(contamination=0.04, random_state=42)
    df['Anomaly'] = iso.fit_predict(df[['Length', 'Cluster_ID']])

    # 4. SMART DUPLICATE DETECTION
    conflicts = []
    recs = df.to_dict('records')
    # Use a sliding window to check for matches efficiently
    for i in range(len(recs)):
        for j in range(i + 1, min(i + 300, len(recs))):
            r1, r2 = recs[i], recs[j]
            
            # Fuzzy Similarity
            sim = SequenceMatcher(None, r1['Clean_Desc'], r2['Clean_Desc']).ratio()
            
            if sim > 0.85:
                # TPM GOLDEN RULE: If Numeric DNA differs, it's a Variant, NOT a duplicate.
                if r1['Num_Fingerprint'] != r2['Num_Fingerprint']:
                    finding = "🛠️ Variant (Distinct Size/Spec)"
                    color = "blue"
                else:
                    finding = "🚨 Potential Duplicate"
                    color = "red"
                
                conflicts.append({
                    'Item A': r1['Item_No'],
                    'Item B': r2['Item_No'],
                    'Desc A': r1['Clean_Desc'],
                    'Desc B': r2['Clean_Desc'],
                    'Similarity': f"{sim:.1%}",
                    'Status': finding,
                    'Color': color
                })
                
    return df, pd.DataFrame(conflicts)

# --- EXECUTION ---
try:
    df, conflicts = run_audit('raw_data.csv')
except:
    st.error("Please upload 'raw_data.csv' to your GitHub repo.")
    st.stop()

# --- DASHBOARD ---
c1, c2, c3 = st.columns(3)
c1.metric("SKUs Audited", len(df))
c2.metric("Anomalies", len(df[df['Anomaly'] == -1]))
c3.metric("Fuzzy Matches Found", len(conflicts))

tab1, tab2, tab3 = st.tabs(["📊 Inventory Categories", "👯 Conflict Manager", "🚨 Data Anomalies"])

with tab1:
    fig = px.bar(df['Category'].value_counts(), title="Top Inventory Groups")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[['Item_No', 'Clean_Desc', 'Category']])

with tab2:
    st.subheader("Duplicate vs. Variant Audit")
    st.info("Our algorithm now distinguishes between text similarity and numeric specification changes.")
    
    if not conflicts.empty:
        # Style the dataframe
        def color_status(val):
            color = 'red' if 'Duplicate' in val else 'blue'
            return f'color: {color}'
        
        st.dataframe(conflicts.drop(columns=['Color']).style.applymap(color_status, subset=['Status']), use_container_width=True)
    else:
        st.success("No conflicts found.")

with tab3:
    st.subheader("Data Pattern Anomalies")
    st.dataframe(df[df['Anomaly'] == -1][['Item_No', 'Clean_Desc', 'Category']])
