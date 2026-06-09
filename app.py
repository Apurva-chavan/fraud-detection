# ============================================================
#  Healthcare Fraud Detection — Streamlit Dashboard (app.py)
# ============================================================
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Healthcare Fraud Detector", page_icon="🏥", layout="wide")

st.title("🏥 Healthcare Provider Fraud Detection")
st.markdown("This dashboard shows fraud predictions made by the ML model.")

OUT    = "outputs/"
CHARTS = "charts/"

# Load submission
sub_path = OUT + "Submission.csv"
if not os.path.exists(sub_path):
    st.error("⚠️ Submission.csv not found. Please run pipeline.py first.")
    st.stop()

df = pd.read_csv(sub_path)

# Top metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Providers", len(df))
col2.metric("Predicted Fraudulent", (df['PredictedClass']=='Yes').sum())
col3.metric("Predicted Legitimate", (df['PredictedClass']=='No').sum())

st.markdown("---")

# Filter
st.subheader("Predictions Table")
filter_opt = st.radio("Filter by:", ["All", "Fraud Only", "Legitimate Only"], horizontal=True)
if filter_opt == "Fraud Only":
    view = df[df['PredictedClass'] == 'Yes']
elif filter_opt == "Legitimate Only":
    view = df[df['PredictedClass'] == 'No']
else:
    view = df

st.dataframe(view.sort_values('Probability', ascending=False).reset_index(drop=True),
             use_container_width=True)

st.markdown("---")

# Charts
st.subheader("Charts")
c1, c2 = st.columns(2)

if os.path.exists(CHARTS + 'fraud_distribution.png'):
    c1.image(CHARTS + 'fraud_distribution.png', caption="Fraud Distribution in Training Data")

if os.path.exists(CHARTS + 'feature_importance.png'):
    c2.image(CHARTS + 'feature_importance.png', caption="Top 20 Important Features")

if os.path.exists(CHARTS + 'confusion_matrix.png'):
    st.image(CHARTS + 'confusion_matrix.png', caption="Confusion Matrix (Validation Set)", width=400)
