# Modified Streamlit App: Signal Processing, Feature Extraction, KMeans Clustering with 3D PCA

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 5})
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import signal, stats
import zipfile
import os
import itertools
import re

st.set_page_config(layout="wide")
st.title("Ultrasonic Signal Feature Clustering")

# --- Extract ZIP ---
def extract_zip(zip_path, extract_dir="ultrasonic_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]

# --- Auto-cropping ---
def auto_crop(data):
    threshold = 0.02 * np.max(np.abs(data))
    active_indices = np.where(np.abs(data) > threshold)[0]
    if len(active_indices) == 0:
        return data
    start_idx = max(0, active_indices[0] - 1000)
    end_idx = min(len(data), active_indices[-1] + 1000)
    return data[start_idx:end_idx]

# --- Feature Extraction ---
def extract_features(data, fs):
    features = {}

    # Time-domain features
    features['mean'] = np.mean(data)
    features['std'] = np.std(data)
    features['var'] = np.var(data)
    features['rms'] = np.sqrt(np.mean(data**2))
    features['skewness'] = stats.skew(data)
    features['kurtosis'] = stats.kurtosis(data)
    features['peak'] = np.max(np.abs(data))
    features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] != 0 else 0
    features['impulse_factor'] = features['peak'] / (np.mean(np.abs(data)) + 1e-12)
    features['shape_factor'] = features['rms'] / (np.mean(np.abs(data)) + 1e-12)

    # Frequency-domain features
    freqs, psd = signal.welch(data, fs=fs)
    features['spectral_centroid'] = np.sum(freqs * psd) / (np.sum(psd) + 1e-12)
    features['spectral_entropy'] = -np.sum((psd / np.sum(psd)) * np.log2(psd / np.sum(psd) + 1e-12))
    features['spectral_flatness'] = stats.gmean(psd + 1e-12) / (np.mean(psd) + 1e-12)
    features['spectral_rolloff'] = freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]

    return features

# --- Sidebar ---
with st.sidebar:
    uploaded_file = st.file_uploader("Upload ZIP of CSVs", type="zip")
    fs = st.number_input("Sampling Frequency (Hz)", min_value=1000, max_value=1000000, value=500000)
    n_clusters = st.number_input("Number of Clusters (K)", min_value=2, max_value=20, value=3)

# --- Main Logic ---
if uploaded_file:
    with open("temp_upload.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_paths = extract_zip("temp_upload.zip")
    all_features = []
    file_labels = []

    for file in file_paths:
        try:
            df = pd.read_csv(file, header=None)
            data = df.iloc[:, 0]  # Always take the first column

            # Extract annotation from filename
            filename = os.path.basename(file)
            try:
                time_str = filename.split("_")[1]
                description_match = re.search(r"Sensor01_(.*?)\.csv", filename)
                label_note = f"{time_str} - {description_match.group(1)}" if description_match else time_str
            except Exception:
                label_note = filename  # fallback if format doesn't match

            cropped = auto_crop(data.values)
            if len(cropped) < 10:
                st.warning(f"Skipping {file}: Too short after cropping.")
                continue

            features = extract_features(cropped, fs)
            all_features.append(features)
            file_labels.append(label_note)
        except Exception as e:
            st.error(f"Error processing {file}: {e}")

    df_features = pd.DataFrame(all_features, index=file_labels)

    if df_features.empty:
        st.error("No features extracted. Check input data or preprocessing steps.")
        st.stop()

    st.subheader("Extracted Features (Raw)")
    st.write(df_features)

    selected_features = st.multiselect("Select features to use for clustering", df_features.columns.tolist(), default=df_features.columns.tolist())

    if selected_features:
        X = df_features[selected_features].values
        X_scaled = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=clusters, cmap='rainbow', s=60)
        ax.set_title("3D PCA Clustering")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        for i, label in enumerate(file_labels):
            ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], label, fontsize=4)
        st.pyplot(fig, use_container_width=False)

        df_features['Cluster'] = clusters
        st.dataframe(df_features)
else:
    st.info("Please upload a ZIP file of single-column CSV files for processing.")
