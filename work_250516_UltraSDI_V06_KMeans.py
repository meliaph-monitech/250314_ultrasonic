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

# --- Session state for persistent parameters ---
if "spec_width" not in st.session_state:
    st.session_state.spec_width = 2.5
if "spec_height" not in st.session_state:
    st.session_state.spec_height = 4.0
if "signal_width" not in st.session_state:
    st.session_state.signal_width = 1.8
if "signal_height" not in st.session_state:
    st.session_state.signal_height = 1.0
if "crop_ranges" not in st.session_state:
    st.session_state.crop_ranges = {}

# --- Function to extract ZIP ---
def extract_zip(zip_path, extract_dir="ultrasonic_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    return [os.path.join(extract_dir, f) for f in csv_files]

def extract_features(signal, fs):
    features = {}

    # Time-domain features
    features["mean"] = np.mean(signal)
    features["std"] = np.std(signal)
    features["max"] = np.max(signal)
    features["min"] = np.min(signal)
    features["skew"] = skew(signal)
    features["kurtosis"] = kurtosis(signal)
    features["rms"] = np.sqrt(np.mean(signal**2))
    features["crest_factor"] = np.max(np.abs(signal)) / features["rms"]

    # # Frequency-domain features
    # freqs, psd = signal_power_spectrum(signal, fs)
    # features["spectral_centroid"] = np.sum(freqs * psd) / np.sum(psd)
    # features["spectral_bandwidth"] = np.sqrt(np.sum((freqs - features["spectral_centroid"])**2 * psd) / np.sum(psd))
    # features["spectral_entropy"] = -np.sum(psd * np.log(psd + 1e-10)) / np.log(len(psd))

    return features

def signal_power_spectrum(signal_data, fs):
    freqs, psd = signal.welch(signal_data, fs=fs)
    psd /= np.sum(psd)  # normalize for entropy
    return freqs, psd


# --- Sidebar: Upload & Parameters ---
with st.sidebar:
    uploaded_file = st.file_uploader("Upload ZIP of CSVs (1-column ultrasonic data)", type="zip")
    st.markdown("### Spectrogram Parameters")
    fs = st.number_input("Sampling Frequency (Hz)", min_value=1000, max_value=500000, value=500000)
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


# --- Main Logic ---
if uploaded_file:
    with open("temp_ultrasonic.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_paths = extract_zip("temp_ultrasonic.zip")
    st.success(f"Extracted {len(file_paths)} CSV files")

    # --- Feature Extraction and K-Means Clustering Across Files ---
    with st.expander("📊 Feature-Based Clustering with PCA (All Files)", expanded=True):
        feature_list = []
        filenames_used = []
    
        for path in file_paths:
            try:
                data = pd.read_csv(path, header=None, usecols=[0], encoding='latin1').squeeze("columns")
    
                # Auto-trim to active signal
                threshold = 0.02 * np.max(np.abs(data))
                active_idx = np.where(np.abs(data) > threshold)[0]
                if len(active_idx) > 0:
                    data = data[max(0, active_idx[0] - 1000): min(len(data), active_idx[-1] + 1000)]
    
                # Time-domain features
                mean_val = np.mean(data)
                std_val = np.std(data)
                max_val = np.max(data)
                min_val = np.min(data)
                rms = np.sqrt(np.mean(np.square(data)))
                energy = np.sum(np.square(data))
                skew = pd.Series(data).skew()
                kurt = pd.Series(data).kurt()
    
                # Frequency-domain features
                freqs, psd = signal.welch(data, fs)
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                spectral_bw = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
    
                feature_vector = [mean_val, std_val, max_val, min_val, rms, energy, skew, kurt, spectral_centroid, spectral_bw]
                feature_list.append(feature_vector)
                filenames_used.append(os.path.basename(path))
    
            except Exception as e:
                st.warning(f"Error in {os.path.basename(path)}: {e}")
    
        if len(feature_list) >= 3:
            X = np.array(feature_list)
            X_scaled = StandardScaler().fit_transform(X)
    
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_scaled)
    
            kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
            labels = kmeans.fit_predict(X_pca)
    
            # 3D scatter plot
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='coolwarm', s=60)
    
            for i, fname in enumerate(filenames_used):
                try:
                    parts = fname.split('_')
                    time_str = parts[1]
                    description = parts[3].replace(".csv", "")
                    label = f"{time_str}_{description}"
                except Exception:
                    label = fname  # fallback if unexpected format
                
                ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], label, fontsize=6)
    
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.set_zlabel("PCA 3")
            ax.set_title("3D PCA + KMeans Clustering")
            st.pyplot(fig)
        else:
            st.warning("Not enough valid files to perform clustering.")
    

    file_labels = [os.path.basename(f) for f in file_paths]
    selected_label = st.selectbox("Select a CSV file to visualize", file_labels)
    file_name = file_paths[file_labels.index(selected_label)]

if file_name:
    raw_data = pd.read_csv(file_name, header=None, usecols=[0], encoding='latin1').squeeze("columns")

    if raw_data.ndim > 1:
        st.error("CSV file has more than one column. Please ensure it's a single-column signal file.")
    else:
        base_name = os.path.basename(file_name).replace(".csv", "").replace("_", " ")
        st.subheader(f"{base_name}")

        # --- Auto-cropping based on signal activity ---
        threshold = 0.02 * np.max(np.abs(raw_data))
        active_indices = np.where(np.abs(raw_data) > threshold)[0]
        if len(active_indices) > 0:
            start_idx = max(0, active_indices[0] - 1000)
            end_idx = min(len(raw_data), active_indices[-1] + 1000)
            raw_data = raw_data[start_idx:end_idx]
            st.info(f"Signal auto-trimmed to region with activity: {len(raw_data)} samples")

        total_duration_ms = int(len(raw_data) / fs * 1000)

        # --- Cropping Time Range Per File ---
        prev_crop = st.session_state.crop_ranges.get(file_name, (0, total_duration_ms))
        crop_start_ms, crop_end_ms = st.slider(
            "Cropping Time Range (ms)",
            min_value=0,
            max_value=total_duration_ms,
            value=prev_crop,
            step=1
        )
        apply_crop = st.button("Apply Cropping")

        if apply_crop:
            st.session_state.crop_ranges[file_name] = (crop_start_ms, crop_end_ms)

        crop_start_idx = int((st.session_state.crop_ranges.get(file_name, (0, total_duration_ms))[0] / 1000) * fs)
        crop_end_idx = int((st.session_state.crop_ranges.get(file_name, (0, total_duration_ms))[1] / 1000) * fs)
        raw_data = raw_data[crop_start_idx:crop_end_idx]
        st.info(f"Cropping applied: {crop_end_idx - crop_start_idx} samples")

        # --- Downsampling if too long ---
        MAX_SAMPLES = 200_000
        if len(raw_data) > MAX_SAMPLES:
            factor = len(raw_data) // MAX_SAMPLES
            raw_data = raw_data[::factor]
            fs = fs // factor
            st.warning(f"Downsampled by {factor}x to reduce memory usage.")

        # --- Raw Signal Plot (appears first) ---
        with st.expander("🔍 Raw Signal Plot", expanded=True):
            time_axis = np.arange(len(raw_data)) / fs * 1000
            fig2, ax2 = plt.subplots(figsize=(st.session_state.signal_width, st.session_state.signal_height))
            ax2.plot(time_axis, raw_data, color='gray')
            ax2.set_xlabel("Time (ms)")
            ax2.set_ylabel("Amplitude")
            ax2.set_title(f"{base_name}")
            st.pyplot(fig2, use_container_width=False)

        # --- Spectrogram ---
        try:
            noverlap = int(noverlap_ratio * nperseg)
            nperseg = min(nperseg, len(raw_data) // 8)
            nfft = min(nfft, 2 ** int(np.floor(np.log2(len(raw_data)))))

            f_vals, t_vals, Sxx = signal.spectrogram(
                raw_data, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='spectrum'
            )
            Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)
            max_dB = np.max(Sxx_dB)
            Sxx_dB[Sxx_dB < max_dB - db_scale] = max_dB - db_scale

            fig, ax = plt.subplots(figsize=(st.session_state.spec_width, st.session_state.spec_height))
            extent = [t_vals[0]*1000, t_vals[-1]*1000, f_vals[0]/1000, f_vals[-1]/1000]
            im = ax.imshow(Sxx_dB, aspect='auto', extent=extent, origin='lower', cmap='jet')
            ax.set_ylim([0, ylimit_khz])
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Frequency (kHz)")
            # ax.set_title(f"{base_name}")
            fig.colorbar(im, ax=ax, label="Intensity (dB)")
            st.pyplot(fig, use_container_width=False)

        except Exception as e:
            st.error(f"Spectrogram computation failed: {e}")
            st.stop()

else:
    st.info("Please upload a ZIP file containing 1-column CSV signal files.")
