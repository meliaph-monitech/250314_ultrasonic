# -*- coding: utf-8 -*-
"""WORK_250314_UltrasonicWeldingSignalEDA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rDcQnhUqmNzPzXHO1Ua4k-FOMkQlfung
"""

import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest

# Function to extract ZIP files
def extract_zip(uploaded_file, extract_to="extracted_data"):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return [os.path.join(extract_to, f) for f in os.listdir(extract_to) if f.endswith('.csv')]

# Function to segment data
def segment_data(file_paths):
    segments = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, header=None)
        signal = df[0].values
        file_name = os.path.basename(file_path)
        threshold = np.std(signal) * 0.1
        is_fluctuating = np.abs(signal) > threshold
        change_points = np.where(np.diff(is_fluctuating.astype(int)) != 0)[0]
        start = 0
        for end in change_points:
            region_type = "fluctuated" if is_fluctuating[start] else "flat"
            segments.append({
                "file_name": file_name,
                "region_type": region_type,
                "region_start_index": start,
                "region_end_index": end
            })
            start = end + 1
    return pd.DataFrame(segments)

# Function to extract features
def extract_features(signal):
    n = len(signal)
    if n == 0:
        return [0] * 20

    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    median_val = np.median(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    peak_to_peak = max_val - min_val
    energy = np.sum(signal**2)
    cv = std_val / mean_val if mean_val != 0 else 0

    # Deactivating FFT and spectral analysis
    # signal_fft = fft(signal)
    # psd = np.abs(signal_fft)**2
    # freqs = fftfreq(n, 1)
    # positive_freqs = freqs[:n // 2]
    # positive_psd = psd[:n // 2]
    # dominant_freq = positive_freqs[np.argmax(positive_psd)] if len(positive_psd) > 0 else 0
    # psd_normalized = positive_psd / np.sum(positive_psd) if np.sum(positive_psd) > 0 else np.zeros_like(positive_psd)
    # spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))

    dominant_freq = 0  # Default value
    spectral_entropy = 0  # Default value

    # Autocorrelation
    autocorrelation = np.corrcoef(signal[:-1], signal[1:])[0, 1] if n > 1 else 0
    peaks, _ = find_peaks(signal)
    peak_count = len(peaks)
    zero_crossing_rate = np.sum(np.diff(np.sign(signal)) != 0) / n
    rms = np.sqrt(np.mean(signal**2))

    # Deactivating Rolling Mean Calculation
    # rolling_window = max(10, n // 10)
    # if n >= rolling_window:
    #     rolling_mean = np.convolve(signal, np.ones(rolling_window) / rolling_window, mode='valid')
    #     moving_average = np.mean(rolling_mean)
    # else:
    #     moving_average = mean_val  # Default to mean if signal is too short
    moving_average = mean_val  # Default value directly

    # Deactivating Outlier Detection and Extreme Event Duration
    # threshold = 3 * std_val
    # outlier_count = np.sum(np.abs(signal - mean_val) > threshold)
    # extreme_event_duration = 0
    # current_duration = 0
    # for value in signal:
    #     if np.abs(value - mean_val) > threshold:
    #         current_duration += 1
    #     else:
    #         extreme_event_duration = max(extreme_event_duration, current_duration)
    #         current_duration = 0
    outlier_count = 0  # Default value
    extreme_event_duration = 0  # Default value

    return [mean_val, std_val, min_val, max_val, median_val, skewness, kurt, peak_to_peak, energy, cv, 
            dominant_freq, spectral_entropy, autocorrelation, peak_count, zero_crossing_rate, rms, 
            0, moving_average, outlier_count, extreme_event_duration]

# Main function
def main():
    st.sidebar.title("Ultrasonic Welding Data Explorer")
    uploaded_file = st.sidebar.file_uploader("Upload ZIP file containing CSVs", type=["zip"])

    if uploaded_file:
        file_paths = extract_zip(uploaded_file)
        st.sidebar.write(f"Total CSV files: {len(file_paths)}")

        if st.sidebar.button("Segment Data"):
            segments_df = segment_data(file_paths)
            st.session_state["segments_df"] = segments_df
            st.sidebar.success("Segmentation Done")
            st.write("## Segmented Data Overview")
            st.dataframe(segments_df)

        if "segments_df" in st.session_state and st.sidebar.button("Begin Feature Extraction"):
            segments_df = st.session_state["segments_df"]
            feature_list = []
            for _, row in segments_df.iterrows():
                file_path = os.path.join("extracted_data", row["file_name"])
                df = pd.read_csv(file_path, header=None)
                signal = df[0].values[row["region_start_index"]:row["region_end_index"]]
                features = extract_features(signal)
                feature_list.append([row["file_name"], row["region_type"]] + features)
            feature_columns = ["file_name", "region_type"] + [f"feature_{i}" for i in range(20)]
            features_df = pd.DataFrame(feature_list, columns=feature_columns)
            st.session_state["features_df"] = features_df
            st.write("## Extracted Features")
            st.dataframe(features_df)
            feature_data = features_df.iloc[:, 2:]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(feature_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(fig)

    # Anomaly detection
    if "features_df" in st.session_state:
        features_df = st.session_state["features_df"]
        st.sidebar.write("## Select Features for Anomaly Detection")
        all_features = features_df.columns[2:]
        selected_features = st.sidebar.multiselect("Choose features:", all_features, default=list(all_features))
        if st.sidebar.button("Start Anomaly Detection"):
            if not selected_features:
                st.sidebar.warning("Please select at least one feature.")
            else:
                X = features_df[selected_features]
                contamination_rate = st.sidebar.slider("Contamination Rate", 0.01, 0.5, 0.05)
                use_contamination = st.sidebar.checkbox("Disable Contamination Rate")
                model = IsolationForest(contamination=contamination_rate if not use_contamination else 'auto', random_state=42)
                anomalies = model.fit_predict(X)
                features_df["anomaly"] = anomalies
                features_df["anomaly"] = features_df["anomaly"].map({1: "normal", -1: "anomaly"})
                st.session_state["features_df"] = features_df
                st.sidebar.success("Anomaly Detection Done!")
                pca = PCA(n_components=3)
                pca_result = pca.fit_transform(X)
                features_df["PCA1"], features_df["PCA2"], features_df["PCA3"] = pca_result[:, 0], pca_result[:, 1], pca_result[:, 2]
                fig = px.scatter_3d(features_df, x="PCA1", y="PCA2", z="PCA3",
                                    color="anomaly",
                                    symbol="region_type",
                                    hover_data=["file_name", "region_type"],
                                    title="3D PCA Scatter Plot")
                st.plotly_chart(fig)

    # Raw signal visualization
    if "features_df" in st.session_state:
        st.write("## Compare Raw Signals")
        anomaly_selection = st.selectbox("Select Data to View:", features_df["file_name"].unique())
        if anomaly_selection:
            selected_data = features_df[features_df["file_name"] == anomaly_selection]
            file_path = os.path.join("extracted_data", anomaly_selection)
            df = pd.read_csv(file_path, header=None)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df[0].values, label="Raw Signal")
            ax.set_title(f"Raw Signal of {anomaly_selection}")
            ax.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
