import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile
import scipy.signal as signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(layout="wide")
st.title("Ultrasonic Welding Signal Clustering Viewer")

# --- Upload ZIP ---
with st.sidebar:
    uploaded_file = st.file_uploader("Upload ZIP of CSVs", type="zip")
    fs = st.number_input("Sampling Frequency (Hz)", value=500000)
    
    st.markdown("### Feature Selection for Clustering")
    all_features = [
        "mean", "std", "max", "min", "rms", "crest_factor",
        "skewness", "kurtosis", "spectral_centroid",
        "spectral_bandwidth", "spectral_entropy"
    ]
    selected_features = st.multiselect("Select features", all_features, default=all_features)

    run_clustering = st.button("Run Clustering")

# --- Helper Functions ---
def extract_zip(zip_path, extract_to="unzipped"):
    if os.path.exists(extract_to):
        for f in os.listdir(extract_to):
            os.remove(os.path.join(extract_to, f))
    else:
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    return [os.path.join(extract_to, f) for f in os.listdir(extract_to) if f.endswith(".csv")]

def extract_features(data, fs):
    features = {}
    features["mean"] = np.mean(data)
    features["std"] = np.std(data)
    features["max"] = np.max(data)
    features["min"] = np.min(data)
    features["rms"] = np.sqrt(np.mean(np.square(data)))
    features["crest_factor"] = np.max(np.abs(data)) / (features["rms"] + 1e-10)
    features["skewness"] = skew(data)
    features["kurtosis"] = kurtosis(data)
    freqs, psd = signal.welch(data, fs)
    psd /= np.sum(psd)
    features["spectral_centroid"] = np.sum(freqs * psd)
    features["spectral_bandwidth"] = np.sqrt(np.sum((freqs - features["spectral_centroid"])**2 * psd))
    features["spectral_entropy"] = -np.sum(psd * np.log(psd + 1e-10)) / np.log(len(psd))
    return features

def extract_label_from_filename(filename):
    parts = filename.split("_")
    if len(parts) >= 4:
        return f"{parts[1]}_{parts[3].replace('.csv','')}"
    return filename

# --- Main Logic ---
if uploaded_file and run_clustering:
    with open("temp.zip", "wb") as f:
        f.write(uploaded_file.read())

    file_paths = extract_zip("temp.zip")
    st.success(f"{len(file_paths)} CSV files extracted.")

    all_feature_vectors = []
    file_labels = []

    for file in file_paths:
        try:
            raw = pd.read_csv(file, header=None).squeeze("columns")
            threshold = 0.02 * np.max(np.abs(raw))
            active = np.where(np.abs(raw) > threshold)[0]
            if len(active) == 0:
                continue
            cropped = raw[max(0, active[0] - 1000): min(len(raw), active[-1] + 1000)]
            feats = extract_features(cropped, fs)
            all_feature_vectors.append([feats[f] for f in selected_features])
            file_labels.append(extract_label_from_filename(os.path.basename(file)))
        except Exception as e:
            st.warning(f"Failed on {file}: {e}")

    if len(all_feature_vectors) >= 3:
        X = np.array(all_feature_vectors)
        X_scaled = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=4, n_init="auto", random_state=0)
        labels = kmeans.fit_predict(X_scaled)

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        # Plotting
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap="tab10", s=60)

        for i, label in enumerate(file_labels):
            ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], label, fontsize=6)

        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("PCA 3")
        ax.set_title("3D PCA Visualization of Clusters")
        st.pyplot(fig)
    else:
        st.warning("Not enough valid files for clustering.")
elif uploaded_file and not run_clustering:
    st.info("Select features and click 'Run Clustering'.")
else:
    st.info("Upload a ZIP file to begin.")
