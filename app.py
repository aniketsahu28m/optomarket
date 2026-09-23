import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import time

api_key = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY')  # Set via env var GEMINI_API_KEY or replace here
genai.configure(api_key=api_key)
GEMINI_KEY_MISSING = (api_key == 'YOUR_API_KEY' or not api_key)

def get_gemini_results(formatted_data, retries=3):
    prompt = f"""
    Here is the detailed customer segmentation data:
    {formatted_data}

    1. Generate a Formal, Professional title for each customer segment. Avoid using puns or captions.
    2. Provide an analysis and detailed summary of each segment, including key characteristics and insights.
    3. Suggest marketing strategies for each segment.
    4. Offer recommendations for targeting and fraud prevention based on segment characteristics.
    5. Do not give any cluster ID. Just say the cluster number and the title.
    6. Create a segment profile in paragraph format with a detailed psychological evaluation of customers in this segment.
    7.Make all parts comprehensive
    8. Start the evaluation directly. DO NOT have an opening line like Here is a detailed analysis of each customer segment, including key characteristics, insights, marketing strategies, targeting recommendations, and fraud prevention measures:

    """

    model = genai.GenerativeModel("gemini-flash-latest")

    attempt = 0
    while attempt < retries:
        try:
            response = model.generate_content(prompt)
            return response.text
        except ResourceExhausted as e:
            attempt += 1
            if attempt < retries:
                wait_time = 2 ** attempt  
                time.sleep(wait_time)
            else:
                print(f"Failed to retrieve results after {retries} attempts: {e}")
                return None

def prepare_gemini_input(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    # Format data as a string for the Gemini API prompt
    formatted_data = json.dumps(data, indent=4)
    return formatted_data

st.set_page_config(page_title="Optomarket - Customer Segmentation", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size:50px;
        color:#3498db;
        text-align:center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size:24px;
        color:#555;
        text-align:center;
        margin-top: -10px;
        font-style: italic;
    }
    .section-title {
        font-size:22px;
        color:#2c3e50;
        margin-bottom:10px;
        font-weight:bold;
    }
    .button-container {
        display: flex;
        justify-content: space-around;
        margin-bottom: 20px;
    }
    .stButton button {
        background-color: #27ae60;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("logo.jpg", width=400)  
st.markdown('<div class="main-title">Optomarket</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI Driven Customer Segmentation and Analysis Tool</div>', unsafe_allow_html=True)




# Sidebar: File Upload
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Initialize session state for data processing
if "data_processed" not in st.session_state:
    st.session_state.data_processed = False

if "current_section" not in st.session_state:
    st.session_state.current_section = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### Data Preview")
    st.write(df.head())

    if st.sidebar.button("Process Data") or st.session_state.data_processed:
        st.session_state.data_processed = True

        # Data preprocessing
        columns_numeric = ['is_fraud', 'customer_id', 'merch_lat', 'merch_long', 'long', 'lat', 'amt', 'city_pop']
        for column in columns_numeric:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            df[column] = df[column].fillna(df[column].median())

        columns_categorical = ['category', 'gender', 'city', 'state', 'job', 'first', 'last', 'merchant']
        for column in columns_categorical:
            df[column] = df[column].fillna(df[column].mode()[0])

        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')

        df['age'] = df.apply(
            lambda row: (row['trans_date_trans_time'].year - row['dob'].year - 
                         ((row['trans_date_trans_time'].month, row['trans_date_trans_time'].day) < 
                          (row['dob'].month, row['dob'].day))) 
            if pd.notnull(row['dob']) and pd.notnull(row['trans_date_trans_time']) 
            else np.nan, axis=1
        )
        df['age'] = df['age'].fillna(df['age'].median())

        # Clustering
        features = ['amt', 'age', 'is_fraud', 'city_pop']
        for feature in features:
            if df[feature].isna().any():
                df[feature] = df[feature].fillna(df[feature].median())
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[features])
        kmeans_model = KMeans(n_clusters=4, random_state=42)
        df['Cluster'] = kmeans_model.fit_predict(df_scaled)

        # Build cluster characteristics from THIS uploaded dataset (so the
        # Recommendations tab always has fresh data to send to Gemini,
        # instead of depending on a separately-generated JSON file).
        cluster_characteristics = df.groupby('Cluster').agg({
            'amt': ['mean', 'std', 'median'],
            'age': ['mean', 'std', 'median'],
            'is_fraud': ['mean', 'std'],
            'city_pop': ['mean', 'std', 'median'],
            'Cluster': 'count'
        }).round(2)
        cluster_characteristics.columns = [
            '_'.join(col).strip() for col in cluster_characteristics.columns.values
        ]
        cluster_characteristics.rename(columns={'Cluster_count': 'Count'}, inplace=True)
        cluster_characteristics = cluster_characteristics.reset_index()

        cluster_json_path = "cluster_characteristics.json"
        with open(cluster_json_path, "w") as f:
            json.dump(cluster_characteristics.to_dict(orient="records"), f, indent=4)

        # UI Buttons for Navigation
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Show Graphs"):
                st.session_state.current_section = 'graphs'
        with col2:
            if st.button("Show Recommendations"):
                st.session_state.current_section = 'recommendations'
        with col3:
            if st.button("View Segmented Data"):
                st.session_state.current_section = 'segmented_data'

        st.markdown('</div>', unsafe_allow_html=True)

        # Display sections based on selected button
        if st.session_state.current_section == 'graphs':
            st.write("### Graphical Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Scatter Plot
                st.write("#### Transaction Amount vs Age by Cluster")
                fig, ax = plt.subplots()
                sns.scatterplot(x='amt', y='age', hue='Cluster', data=df, palette='coolwarm', ax=ax)
                st.pyplot(fig)
                st.caption("Scatter plot displaying how transaction amount varies with customer age across different clusters.")

            with col2:
                # Violin Plot
                st.write("#### Violin Plot of Transaction Amount by Cluster")
                fig, ax = plt.subplots()
                sns.violinplot(x='Cluster', y='amt', data=df, palette='coolwarm', ax=ax)
                st.pyplot(fig)
                st.caption("Violin plot showing the distribution of transaction amounts for each cluster.")

            col3, col4 = st.columns(2)

            with col3:
                # Histogram for Age
                st.write("#### Distribution of Age across Clusters")
                fig, ax = plt.subplots()
                for cluster in df['Cluster'].unique():
                    sns.histplot(df[df['Cluster'] == cluster]['age'], kde=True, label=f'Cluster {cluster}', ax=ax)
                ax.legend()
                st.pyplot(fig)
                st.caption("Histogram depicting the age distribution across customer clusters.")

            with col4:
                # PCA Plot
                st.write("#### PCA Plot for Clusters")
                pca = PCA(n_components=2)
                df_pca = pca.fit_transform(df_scaled)
                fig, ax = plt.subplots()
                scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['Cluster'], cmap='coolwarm', alpha=0.6)
                plt.colorbar(scatter, label='Cluster')
                st.pyplot(fig)
                st.caption("Principal Component Analysis (PCA) reducing customer features into 2D, visualized by cluster.")

        elif st.session_state.current_section == 'recommendations':
            st.write("### Segment Analysis and Marketing Recommendations")
            sample_size = min(1000, len(df))  
            segment_json = df.sample(sample_size).to_json(orient='records')
            if GEMINI_KEY_MISSING:
                st.warning(
                    "No Gemini API key detected. Set the GEMINI_API_KEY environment "
                    "variable before running the app to enable AI-generated segment "
                    "insights."
                )
            elif os.path.exists(cluster_json_path):
                formatted_data = prepare_gemini_input(cluster_json_path)
                with st.spinner("Asking Gemini for segment insights..."):
                    analysis = get_gemini_results(formatted_data)
                if analysis:
                    st.write(analysis)
                else:
                    st.error(
                        "Gemini did not return a result. Check that GEMINI_API_KEY "
                        "is set correctly and that you have available quota."
                    )
            else:
                st.error(
                    "cluster_characteristics.json was not found. Click 'Process Data' "
                    "again to regenerate it."
                )

            st.download_button("Download Segment Profiles", data=segment_json, file_name="segment_profiles.json", mime="application/json")

        elif st.session_state.current_section == 'segmented_data':
            st.write("### Segmented Data Samples")
            for cluster_label in df['Cluster'].unique():
                st.write(f"#### Segment {cluster_label + 1} Preview ") 
                st.write(df[df['Cluster'] == cluster_label].head())
            st.download_button("Download Segmented Data", data=df.to_csv(index=False), file_name="segmented_data.csv", mime="text/csv")

else:
    st.write("Please upload a CSV file to begin customer analysis.")

