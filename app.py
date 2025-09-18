import streamlit as st
import pandas as pd
import numpy as np
import joblib

#Load Models and Data
@st.cache_resource
def load_models():
    try:
        clustering_model = joblib.load("rfm_kmeans.pkl")
        scaler = joblib.load("scaler.pkl")
        product_similarity = pd.read_pickle("product_similarity.pkl")  #pandas for DataFrame
        data = pd.read_csv("online_retail.csv")
        return clustering_model, scaler, product_similarity, data
    except Exception as e:
        st.error(f"Error loading models/data: {e}")
        return None, None, None, None

clustering_model, scaler, product_similarity, data = load_models()

#Helper Functions
def recommend_products(product_name, similarity_matrix, data, top_n=5):
    if product_name not in similarity_matrix.index:
        return []
    # Get similarity scores
    scores=similarity_matrix.loc[product_name].sort_values(ascending=False)
    recommendations=[p for p in scores.index if p != product_name][:top_n]
    return recommendations

def predict_cluster(r, f, m, scaler, model):
    try:
        X = np.array([[r, f, m]])
        X_scaled = scaler.transform(X)
        cluster_id = model.predict(X_scaled)[0]
        #Map cluster to label
        cluster_labels = {
            0: "High-Value",
            1: "Regular",
            2: "Occasional",
            3: "At-Risk"
        }
        return cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
    except Exception as e:
        return f"Error: {e}"

#Streamlit UI
st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")
st.title("🛒 Shopper Spectrum: Customer Segmentation & Product Recommendations")

#Product Recommendation
st.header("🎯 Product Recommendation")
product_name = st.text_input("Enter a product name:")
if st.button("Get Recommendations"):
    if product_name.strip() == "":
        st.error("Please enter a product name.")
    elif product_similarity is None or data is None:
        st.error("Recommendation system not loaded properly.")
    else:
        recs = recommend_products(product_name, product_similarity, data)
        if not recs:
            st.error("Product not found in database.")
        else:
            st.success("Top Recommendations:")
            for i, rec in enumerate(recs, 1):
                st.write(f"{i}. {rec}")

#Customer Segmentation
st.header("🔍 Customer Segmentation")
col1, col2, col3=st.columns(3)
with col1:
    recency=st.number_input("Recency (days)", min_value=0, step=1)
with col2:
    frequency=st.number_input("Frequency (# purchases)", min_value=0, step=1)
with col3:
    monetary=st.number_input("Monetary (total spend)", min_value=0.0, step=10.0)

if st.button("Predict Cluster"):
    if clustering_model is None or scaler is None:
        st.error("Clustering model not loaded properly.")
    else:
        label = predict_cluster(recency, frequency, monetary, scaler, clustering_model)
        st.success(f"Predicted Segment: **{label}**")