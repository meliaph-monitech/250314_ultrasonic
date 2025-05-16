import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import zipfile
import os

st.set_page_config(layout="wide")
st.title("Ultrasonic Welding Signal Spectrogram Viewer")

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

# --- Sidebar: Upload & Parameters ---
with st.sidebar:
    uploaded_file = st.file_uploader("Upload ZIP of CSVs (1-column ultrasonic data)", type="zip")
    fs = st.number_input("Sampling Frequency (Hz)", min_value=1000, max_value=100000, value=40000)
    nperseg = st.number_input("nperseg", min_value=64, max_value=8192, value=1024)
    noverlap_ratio = st.slider("Overlap Ratio", min_value=0.0, max_value=0.99, value=0.9)
    nfft = st.number_input("nfft", min_value=256, max_value=16384, value=2048)
    db_scale = st.number_input("dB Dynamic Range", min_value=20, max_value=150, value=100)
    ylimit = st.number_input("Max Frequency Display (Hz)", min_value=1000, max_value=int(fs / 2), value=10000)

# --- Main Logic ---
if uploaded_file:
    with open("temp_ultrasonic.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_paths = extract_zip("temp_ultrasonic.zip")
    st.success(f"Extracted {len(file_paths)} CSV files")

    file_name = st.selectbox("Select a CSV file to visualize", file_paths)

    if file_name:
        raw_data = pd.read_csv(file_name, header=None).squeeze("columns")
        if raw_data.ndim > 1:
            st.error("CSV file has more than one column. Please ensure it's a single-column signal file.")
        else:
            st.subheader(f"Spectrogram for: {os.path.basename(file_name)}")

            noverlap = int(noverlap_ratio * nperseg)
            f_vals, t_vals, Sxx = signal.spectrogram(raw_data, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)
            max_dB = np.max(Sxx_dB)
            Sxx_dB[Sxx_dB < max_dB - db_scale] = max_dB - db_scale

            fig, ax = plt.subplots(figsize=(10, 4))
            pcm = ax.pcolormesh(t_vals, f_vals, Sxx_dB, shading='gouraud', cmap='jet')
            ax.set_ylim([0, ylimit])
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (s)")
            fig.colorbar(pcm, ax=ax, label="Intensity (dB)")
            st.pyplot(fig)

            with st.expander("Raw Signal Plot"):
                fig2, ax2 = plt.subplots(figsize=(10, 2))
                time_axis = np.arange(len(raw_data)) / fs
                ax2.plot(time_axis, raw_data, color='gray')
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Amplitude")
                st.pyplot(fig2)
else:
    st.info("Please upload a ZIP file containing 1-column CSV signal files.")
