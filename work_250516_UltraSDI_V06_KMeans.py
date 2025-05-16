import streamlit as st
import numpy as np
import pandas as pd
import os
import zipfile
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Streamlit config ---
st.set_page_config(layout="wide")
st.title("Ultrasonic Signal Clustering with PCA")

# --- Sidebar Parameters ---
with st.sidebar:
    uploaded_file = st.file_uploader("Upload ZIP of 1-column CSVs", type="zip")
    
    fs = st.number_input("Sampling Frequency (Hz)", min_value=1000, max_value=500000, value=500000)
    
    st.markdown("### Select Features for Clustering")
    available_features = [
        "mean", "std", "max", "min", "rms",
        "skewness", "kurtosis", "crest_factor",
        "spectral_centroid", "spectral_bandwidth", "spectral_entropy"
    ]
    selected_features = st.multiselect("Features", available_features, default=available_features)
    
    run_cluster = st.button("Run Clustering")

# --- Utility Functions ---
def extract_zip(zip_path, extract_dir="ultrasonic_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith(".csv")]

def compute_features(signal_data, fs):
    features = {}
    signal_data = signal_data - np.mean(signal_data)

    # Time-domain
    features["mean"] = np.mean(signal_data)
    features["std"] = np.std(signal_data)
    features["max"] = np.max(signal_data)
    features["min"] = np.min(signal_data)
    features["rms"] = np.sqrt(np.mean(signal_data**2))
    features["skewness"] = skew(signal_data)
    features["kurtosis"] = kurtosis(signal_data)
    features["crest_factor"] = np.max(np.abs(signal_data)) / (features["rms"] + 1e-6)

    # Frequency-domain
    freqs, psd = signal.welch(signal_data, fs=fs)
    psd_norm = psd / (np.sum(psd) + 1e-10)
    features["spectral_centroid"] = np.sum(freqs * psd_norm)
    features["spectral_bandwidth"] = np.sqrt(np.sum(((freqs - features["spectral_centroid"]) ** 2) * psd_norm))
    features["spectral_entropy"] = -np.sum(psd_norm * np.log(psd_norm + 1e-10)) / np.log(len(psd_norm))

    return features

# --- Main Clustering Logic ---
if uploaded_file and run_cluster:
    with open("temp.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    file_paths = extract_zip("temp.zip")
    all_features = []
    labels = []
    
    for path in file_paths:
        try:
            data = pd.read_csv(path, header=None).squeeze("columns")
            if data.ndim != 1:
                continue

            # Auto-trim signal
            threshold = 0.02 * np.max(np.abs(data))
            active_idx = np.where(np.abs(data) > threshold)[0]
            if len(active_idx) > 0:
                data = data[max(0, active_idx[0] - 1000): min(len(data), active_idx[-1] + 1000)]
            else:
                continue

            features = compute_features(data, fs)
            selected_vector = [features[f] for f in selected_features if f in features]
            all_features.append(selected_vector)
            labels.append(os.path.basename(path))

        except Exception as e:
            st.warning(f"Skipping file {path}: {e}")

    if len(all_features) >= 3:
        X = np.array(all_features)
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
        y_kmeans = kmeans.fit_predict(X_pca)

        # Plot 3D PCA
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_kmeans, cmap='tab10', s=60)

        for i, label in enumerate(labels):
            try:
                parts = label.split("_")
                time_str = parts[1]
                desc = parts[3].replace(".csv", "")
                tag = f"{time_str}_{desc}"
            except Exception:
                tag = label
            ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], tag, fontsize=6)

        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("PCA 3")
        ax.set_title("K-Means Clustering (3D PCA)")
        st.pyplot(fig)
    else:
        st.warning("Not enough valid signals to cluster (minimum 3).")
elif uploaded_file and not run_cluster:
    st.info("Upload complete. Click 'Run Clustering' to begin.")
else:
    st.info("Please upload a ZIP file containing 1-column CSV files.")
