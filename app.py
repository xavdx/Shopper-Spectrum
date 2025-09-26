import streamlit as st
import pandas as pd
import numpy as np
import joblib
#Load Models and Data
@st.cache_resource
def load_models():
    try:
        clustering_model=joblib.load("rfm_kmeans3.pkl")
        scaler=joblib.load("scaler3.pkl")
        product_similarity=pd.read_pickle("product_similarity3.pkl")  #DataFrame
        data=pd.read_csv("online_retail.csv")
        return clustering_model, scaler, product_similarity, data
    except Exception as e:
        st.error(f"Error loading models/data: {e}")
        return None, None, None, None

clustering_model, scaler, product_similarity, data=load_models()
#Helper Functions
def recommend_products(product_name, similarity_matrix, data, top_n=5):
    """Return top-N similar products based on similarity matrix"""
    if product_name not in similarity_matrix.index:
        return []
    scores=similarity_matrix.loc[product_name].sort_values(ascending=False)
    recommendations=[p for p in scores.index if p != product_name][:top_n]
    return recommendations

def get_cluster_summary(model, scaler):
    """Return cluster centers in original RFM scale"""
    centers=scaler.inverse_transform(model.cluster_centers_)
    cluster_summary=pd.DataFrame(
        centers, columns=["Recency", "Frequency", "Monetary"]
    )
    cluster_summary["Cluster"]=cluster_summary.index
    return cluster_summary

def assign_segment(r, f, m, scaler, model, cluster_summary):
    """Assign segment dynamically based on RFM relative to cluster centers"""
    X=np.array([[r, f, m]])
    X_scaled=scaler.transform(X)
    cluster_id=model.predict(X_scaled)[0]
    center=cluster_summary.loc[cluster_id]

    if (f >= center["Frequency"]) and (m >= center["Monetary"]):
        return "High-Value"
    elif (f < center["Frequency"]) and (m >= center["Monetary"]):
        return "Occasional"
    elif (f >= center["Frequency"]) and (m < center["Monetary"]):
        return "Regular"
    else:
        return "At-Risk"
#Streamlit UI
st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")
st.title("🛒Shopper Spectrum: Customer Segmentation & Product Recommendations")

#Product Recommendation
st.header("Product Recommendation")
product_name=st.text_input("Enter a product name:")
if st.button("Get Recommendations"):
    if product_name.strip() == "":
        st.error("Please enter a product name.")
    elif product_similarity is None or data is None:
        st.error("Recommendation system not loaded properly.")
    else:
        recs=recommend_products(product_name, product_similarity, data)
        if not recs:
            st.error("Product not found in database.")
        else:
            st.success("Top Recommendations:")
            for i, rec in enumerate(recs, 1):
                st.write(f"{i}. {rec}")


#Customer Segmentation
st.header("Customer Segmentation")
col1, col2, col3=st.columns(3)
with col1:
    recency=st.number_input("Recency (days since last purchase)", min_value=0, step=1)
with col2:
    frequency=st.number_input("Frequency (number of purchases)", min_value=0, step=1)
with col3:
    monetary=st.number_input("Monetary (total spend)", min_value=0.0, step=10.0)

if st.button("Predict Cluster"):
    if clustering_model is None or scaler is None:
        st.error("Clustering model not loaded properly.")
    else:
        cluster_summary=get_cluster_summary(clustering_model, scaler)
        label=assign_segment(
            recency, frequency, monetary, scaler, clustering_model, cluster_summary
        )
        st.success(f"Predicted Segment: **{label}**")