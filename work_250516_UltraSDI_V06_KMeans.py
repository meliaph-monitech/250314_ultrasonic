import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7})
import scipy.signal as signal
from scipy.stats import skew, kurtosis
import zipfile
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(layout="wide")
st.title("Ultrasonic Welding Signal Spectrogram Viewer")

# --- Session state defaults ---
def init_session_state():
    defaults = {
        "spec_width": 2.5,
        "spec_height": 4.0,
        "signal_width": 1.8,
        "signal_height": 1.0,
        "crop_ranges": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_session_state()

# --- Helper functions ---
def extract_zip(zip_path, extract_dir="ultrasonic_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]

def extract_features(signal_data, fs):
    features = {
        "mean": np.mean(signal_data),
        "std": np.std(signal_data),
        "max": np.max(signal_data),
        "min": np.min(signal_data),
        "rms": np.sqrt(np.mean(signal_data ** 2)),
        "crest_factor": np.max(np.abs(signal_data)) / (np.sqrt(np.mean(signal_data ** 2)) + 1e-12),
        "skewness": skew(signal_data),
        "kurtosis": kurtosis(signal_data),
    }
    freqs, psd = signal.welch(signal_data, fs=fs)
    psd /= np.sum(psd)
    features.update({
        "spectral_entropy": -np.sum(psd * np.log(psd + 1e-10)) / np.log(len(psd)),
        "spectral_centroid": np.sum(freqs * psd),
        "bandwidth": np.sqrt(np.sum((freqs - np.sum(freqs * psd))**2 * psd))
    })
    return features

# --- Sidebar ---
with st.sidebar:
    uploaded_file = st.file_uploader("Upload ZIP of CSVs (1-column ultrasonic data)", type="zip")
    fs = st.number_input("Sampling Frequency (Hz)", min_value=1000, max_value=500000, value=500000)
    st.markdown("### Spectrogram Parameters")
    nperseg = st.number_input("nperseg", min_value=64, max_value=8192, value=1024)
    noverlap_ratio = st.slider("Overlap Ratio", min_value=0.0, max_value=0.99, value=0.99)
    nfft = st.number_input("nfft", min_value=256, max_value=16384, value=2048)
    db_scale = st.number_input("dB Dynamic Range", min_value=20, max_value=500, value=250)
    ylimit_khz = st.number_input("Max Frequency Display (kHz)", min_value=1, max_value=int(fs / 2000), value=250)
    st.markdown("### Plot Dimensions")
    st.session_state.signal_width = st.slider("Signal Plot Width", 1.0, 10.0, st.session_state.signal_width, step=0.1)
    st.session_state.signal_height = st.slider("Signal Plot Height", 1.0, 10.0, st.session_state.signal_height, step=0.1)
    st.session_state.spec_width = st.slider("Spectrogram Width", 1.0, 10.0, st.session_state.spec_width, step=0.1)
    st.session_state.spec_height = st.slider("Spectrogram Height", 1.0, 10.0, st.session_state.spec_height, step=0.1)
    run_clustering = st.button("Run Clustering")

# --- Main logic ---
if uploaded_file:
    with open("temp_ultrasonic.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_paths = extract_zip("temp_ultrasonic.zip")
    st.success(f"Extracted {len(file_paths)} CSV files")

    if run_clustering:
        st.subheader("📊 Feature-Based Clustering with PCA")
        feature_vectors, filenames = [], []

        for path in file_paths:
            try:
                data = pd.read_csv(path, header=None, usecols=[0], encoding='latin1').squeeze("columns")
                threshold = 0.02 * np.max(np.abs(data))
                active_idx = np.where(np.abs(data) > threshold)[0]
                if len(active_idx):
                    data = data[max(0, active_idx[0]-1000):min(len(data), active_idx[-1]+1000)]
                feats = extract_features(data, fs)
                feature_vectors.append([feats[k] for k in feats])
                filenames.append(os.path.basename(path))
            except Exception as e:
                st.warning(f"Failed {os.path.basename(path)}: {e}")

        if len(feature_vectors) >= 3:
            X = StandardScaler().fit_transform(feature_vectors)
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X)
            labels = KMeans(n_clusters=4, random_state=0, n_init='auto').fit_predict(X_pca)

            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='tab10', s=60)
            for i, fname in enumerate(filenames):
                ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], fname[:10], fontsize=6)
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.set_zlabel("PCA 3")
            st.pyplot(fig)
        else:
            st.warning("Not enough files for clustering.")
