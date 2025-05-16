import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import skew, kurtosis
import zipfile
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Set Streamlit page config
st.set_page_config(layout="wide")
st.title("Ultrasonic Welding Signal Spectrogram Viewer")

# --- Session State Defaults ---
for key, default in {
    "spec_width": 2.5,
    "spec_height": 4.0,
    "signal_width": 1.8,
    "signal_height": 1.0,
    "crop_ranges": {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Helper Functions ---
def extract_zip(zip_path, extract_dir="ultrasonic_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]

def signal_power_spectrum(signal_data, fs):
    freqs, psd = signal.welch(signal_data, fs=fs)
    psd /= np.sum(psd)
    return freqs, psd

def extract_features(signal_data, fs):
    features = {
        "mean": np.mean(signal_data),
        "std": np.std(signal_data),
        "max": np.max(signal_data),
        "min": np.min(signal_data),
        "rms": np.sqrt(np.mean(signal_data**2)),
        "crest_factor": np.max(np.abs(signal_data)) / (np.sqrt(np.mean(signal_data**2)) + 1e-10),
        "skew": skew(signal_data),
        "kurtosis": kurtosis(signal_data)
    }
    freqs, psd = signal_power_spectrum(signal_data, fs)
    features.update({
        "spectral_centroid": np.sum(freqs * psd) / (np.sum(psd) + 1e-10),
        "spectral_bandwidth": np.sqrt(np.sum((freqs - features["spectral_centroid"])**2 * psd) / (np.sum(psd) + 1e-10)),
        "spectral_entropy": -np.sum(psd * np.log(psd + 1e-10)) / np.log(len(psd))
    })
    return features

# --- Sidebar Controls ---
with st.sidebar:
    uploaded_file = st.file_uploader("Upload ZIP of CSVs (1-column ultrasonic data)", type="zip")
    fs = st.number_input("Sampling Frequency (Hz)", min_value=1000, max_value=500000, value=500000)
    nperseg = st.number_input("nperseg", min_value=64, max_value=8192, value=1024)
    noverlap_ratio = st.slider("Overlap Ratio", min_value=0.0, max_value=0.99, value=0.99)
    nfft = st.number_input("nfft", min_value=256, max_value=16384, value=2048)
    db_scale = st.number_input("dB Dynamic Range", min_value=20, max_value=500, value=250)
    ylimit_khz = st.number_input("Max Frequency Display (kHz)", min_value=1, max_value=int(fs / 2000), value=250)
    st.session_state.signal_width = st.slider("Signal Plot Width", 1.0, 10.0, st.session_state.signal_width)
    st.session_state.signal_height = st.slider("Signal Plot Height", 1.0, 10.0, st.session_state.signal_height)
    st.session_state.spec_width = st.slider("Spectrogram Width", 1.0, 10.0, st.session_state.spec_width)
    st.session_state.spec_height = st.slider("Spectrogram Height", 1.0, 10.0, st.session_state.spec_height)
    available_features = [
        "mean", "std", "max", "min", "rms", "crest_factor",
        "skew", "kurtosis", "spectral_centroid",
        "spectral_bandwidth", "spectral_entropy"
    ]
    selected_features = st.multiselect("Select features for clustering", available_features, default=available_features)
    run_cluster = st.button("Run Clustering")

# --- Main Logic ---
if uploaded_file:
    with open("temp_ultrasonic.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_paths = extract_zip("temp_ultrasonic.zip")
    st.success(f"Extracted {len(file_paths)} CSV files")

    # --- Clustering Block ---
    if run_cluster:
        feature_list, filenames_used = [], []
        for path in file_paths:
            try:
                data = pd.read_csv(path, header=None, usecols=[0]).squeeze("columns")
                threshold = 0.02 * np.max(np.abs(data))
                active_idx = np.where(np.abs(data) > threshold)[0]
                if len(active_idx) > 0:
                    data = data[max(0, active_idx[0] - 1000): min(len(data), active_idx[-1] + 1000)]
                feats = extract_features(data, fs)
                vec = [feats[feat] for feat in selected_features if feat in feats]
                feature_list.append(vec)
                filenames_used.append(os.path.basename(path))
            except Exception as e:
                st.warning(f"{os.path.basename(path)}: {e}")

        if len(feature_list) >= 3:
            X_scaled = StandardScaler().fit_transform(np.array(feature_list))
            kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
            labels = kmeans.fit_predict(X_scaled)
            pca = PCA(n_components=3).fit_transform(X_scaled)

            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c=labels, cmap='tab10', s=60)

            for i, fname in enumerate(filenames_used):
                parts = fname.split('_')
                label = f"{parts[1]}_{parts[3].replace('.csv','')}" if len(parts) > 3 else fname
                ax.text(pca[i, 0], pca[i, 1], pca[i, 2], label, fontsize=6)

            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.set_zlabel("PCA 3")
            ax.set_title("3D PCA + KMeans Clustering")
            st.pyplot(fig)
        else:
            st.warning("Not enough valid files to cluster.")
