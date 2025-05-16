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
    fs = st.number_input("Sampling Frequency (Hz)", min_value=1000, max_value=500000, value=500000)
    nperseg = st.number_input("nperseg", min_value=64, max_value=8192, value=1024)
    noverlap_ratio = st.slider("Overlap Ratio", min_value=0.0, max_value=0.99, value=0.9)
    nfft = st.number_input("nfft", min_value=256, max_value=16384, value=2048)
    db_scale = st.number_input("dB Dynamic Range", min_value=20, max_value=500, value=250)
    ylimit_khz = st.number_input("Max Frequency Display (kHz)", min_value=1, max_value=int(fs / 2000), value=250)

# --- Main Logic ---
if uploaded_file:
    with open("temp_ultrasonic.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_paths = extract_zip("temp_ultrasonic.zip")
    st.success(f"Extracted {len(file_paths)} CSV files")

    file_name = st.selectbox("Select a CSV file to visualize", file_paths)

    if file_name:
        raw_data = pd.read_csv(file_name, header=None, usecols=[0], encoding='latin1').squeeze("columns")

        if raw_data.ndim > 1:
            st.error("CSV file has more than one column. Please ensure it's a single-column signal file.")
        else:
            st.subheader(f"Spectrogram for: {os.path.basename(file_name)}")

            # --- Auto-cropping based on signal activity ---
            threshold = 0.01 * np.max(np.abs(raw_data))
            active_indices = np.where(np.abs(raw_data) > threshold)[0]
            if len(active_indices) > 0:
                start_idx = max(0, active_indices[0] - 1000)
                end_idx = min(len(raw_data), active_indices[-1] + 1000)
                raw_data = raw_data[start_idx:end_idx]
                st.info(f"Signal auto-trimmed to region with activity: {len(raw_data)} samples")

            total_duration_ms = int(len(raw_data) / fs * 1000)

            # --- Sidebar cropping control ---
            with st.sidebar:
                st.markdown("### Cropping Time Range (ms)")
                crop_start_ms, crop_end_ms = st.slider(
                    "Select time range (ms)",
                    min_value=0,
                    max_value=total_duration_ms,
                    value=(0, total_duration_ms),
                    step=1
                )
                do_crop = st.button("Revisualize")

            if do_crop:
                crop_start_idx = int((crop_start_ms / 1000) * fs)
                crop_end_idx = int((crop_end_ms / 1000) * fs)
                raw_data = raw_data[crop_start_idx:crop_end_idx]
                st.info(f"Cropping applied: {crop_end_ms - crop_start_ms} ms window.")

            # --- Downsampling if too long ---
            MAX_SAMPLES = 100_000
            if len(raw_data) > MAX_SAMPLES:
                factor = len(raw_data) // MAX_SAMPLES
                raw_data = raw_data[::factor]
                fs = fs // factor
                st.warning(f"Downsampled by {factor}x to reduce memory usage.")

            # --- Spectrogram Calculation ---
            noverlap = int(noverlap_ratio * nperseg)
            nperseg = min(nperseg, len(raw_data) // 8)
            nfft = min(nfft, 2 ** int(np.floor(np.log2(len(raw_data)))))

            try:
                f_vals, t_vals, Sxx = signal.spectrogram(
                    raw_data, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='spectrum'
                )
                Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)
                max_dB = np.max(Sxx_dB)
                Sxx_dB[Sxx_dB < max_dB - db_scale] = max_dB - db_scale

                fig, ax = plt.subplots(figsize=(8, 3))
                # extent = [t_vals[0], t_vals[-1], f_vals[0] / 1000, f_vals[-1] / 1000]  # kHz
                extent = [t_vals[0] * 1000, t_vals[-1] * 1000, f_vals[0] / 1000, f_vals[-1] / 1000]  # Time in ms, Frequency in kHz
                
                im = ax.imshow(Sxx_dB, aspect='auto', extent=extent, origin='lower', cmap='jet')
                ax.set_ylim([0, ylimit_khz])
                # ax.set_xlabel("Time (s)")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Frequency (kHz)")
                fig.colorbar(im, ax=ax, label="Intensity (dB)")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Spectrogram computation failed: {e}")
                st.stop()

            with st.expander("Raw Signal Plot"):
                fig2, ax2 = plt.subplots(figsize=(10, 2))
                # time_axis = np.arange(len(raw_data)) / fs
                time_axis = np.arange(len(raw_data)) / fs
                
                ax2.plot(time_axis, raw_data, color='gray')
                # ax2.set_xlabel("Time (s)")
                ax2.set_xlabel("Time (ms)")
                ax2.set_ylabel("Amplitude")
                st.pyplot(fig2)
else:
    st.info("Please upload a ZIP file containing 1-column CSV signal files.")
