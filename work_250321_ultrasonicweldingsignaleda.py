# -*- coding: utf-8 -*-
import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis

# Function to extract ZIP files
def extract_zip(uploaded_file, extract_to="extracted_data"):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return [os.path.join(extract_to, f) for f in os.listdir(extract_to) if f.endswith('.csv')]

# Function to automatically determine the threshold
def determine_threshold(signal_data):
    """
    Automatically determines the threshold for welding phase separation
    using the interquartile range (IQR) method.
    """
    q1 = np.percentile(signal_data, 25)  # First quartile
    q3 = np.percentile(signal_data, 75)  # Third quartile
    iqr = q3 - q1  # Interquartile range
    return q3 + 1.5 * iqr  # Upper bound for outliers

# Function to separate welding phases
def separate_welding_phases(signal_data):
    if signal_data.empty:  # Handle empty signal data
        return []
    
    threshold = determine_threshold(signal_data)  # Automatically determine threshold
    welding_mask = np.abs(signal_data) > threshold
    welding_phases = []

    # Identify welding phase start and end indices
    change_points = np.where(np.diff(welding_mask.astype(int)) != 0)[0] + 1
    if welding_mask.iloc[0]:  # If signal starts in a welding phase
        change_points = np.insert(change_points, 0, 0)
    if welding_mask.iloc[-1]:  # If signal ends in a welding phase
        change_points = np.append(change_points, len(signal_data))

    # Extract welding phases (only first two are considered)
    for i in range(0, len(change_points) - 1, 2):
        if len(welding_phases) < 2:  # Limit to two welding phases
            welding_phases.append(signal_data.iloc[change_points[i]:change_points[i + 1]])
        else:
            break

    return welding_phases

# Function to extract features
def extract_features(signal):
    n = len(signal)
    if n == 0:
        return [0] * 10

    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    peak_to_peak = max_val - min_val
    energy = np.sum(signal**2)
    rms = np.sqrt(np.mean(signal**2))
    zero_crossing_rate = np.sum(np.diff(np.sign(signal)) != 0) / n

    return [mean_val, std_val, min_val, max_val, skewness, kurt, peak_to_peak, energy, rms, zero_crossing_rate]

# Main function
def main():
    st.sidebar.title("Ultrasonic Welding Data Explorer")
    uploaded_file = st.sidebar.file_uploader("Upload ZIP file containing CSVs", type=["zip"])

    if uploaded_file:
        file_paths = extract_zip(uploaded_file)
        st.sidebar.write(f"Total CSV files: {len(file_paths)}")

        if st.sidebar.button("Segment Welding Phases"):
            segments = []
            for file_path in file_paths:
                try:
                    df = pd.read_csv(file_path, header=None)
                    signal = df[0]
                    file_name = os.path.basename(file_path)

                    # Separate welding phases
                    welding_phases = separate_welding_phases(signal)
                    for i, phase in enumerate(welding_phases):
                        segments.append({
                            "file_name": file_name,
                            "phase_id": i + 1,
                            "signal": phase
                        })
                except Exception as e:
                    st.error(f"Error processing file {file_path}: {e}")
                    continue

            st.session_state["segments"] = segments
            st.sidebar.success("Welding phases segmented successfully!")

            # Visualization of segmentation
            st.write("## Segmented Welding Phases Visualization")
            for file_path in file_paths:
                try:
                    df = pd.read_csv(file_path, header=None)
                    signal = df[0]
                    file_name = os.path.basename(file_path)

                    # Plot the original signal with segmented phases
                    plt.figure(figsize=(12, 6))
                    plt.plot(signal, label="Original Signal", color="black", alpha=0.7)

                    # Highlight segmented welding phases
                    welding_phases = separate_welding_phases(signal)
                    for i, phase in enumerate(welding_phases):
                        start_idx = phase.index[0]
                        end_idx = phase.index[-1]
                        plt.axvspan(start_idx, end_idx, color="blue" if i == 0 else "green", alpha=0.3, label=f"Phase {i + 1}")

                    plt.title(f"Signal Segmentation for {file_name}")
                    plt.xlabel("Time")
                    plt.ylabel("Signal Value")
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt)
                except Exception as e:
                    st.error(f"Error visualizing file {file_name}: {e}")

        if "segments" in st.session_state and st.sidebar.button("Extract Features"):
            segments = st.session_state["segments"]
            feature_list = []
            for segment in segments:
                features = extract_features(segment["signal"])
                feature_list.append([segment["file_name"], segment["phase_id"]] + features)

            feature_columns = ["file_name", "phase_id"] + [f"feature_{i}" for i in range(10)]
            features_df = pd.DataFrame(feature_list, columns=feature_columns)
            st.session_state["features_df"] = features_df
            st.write("## Extracted Features")
            st.dataframe(features_df)

        if "features_df" in st.session_state and st.sidebar.button("Perform Isolation Forest"):
            features_df = st.session_state["features_df"]
            selected_features = features_df.iloc[:, 2:]

            # Perform Isolation Forest
            model = IsolationForest(contamination=0.05, random_state=42)
            anomalies = model.fit_predict(selected_features)
            features_df["anomaly"] = anomalies
            features_df["anomaly"] = features_df["anomaly"].map({1: "normal", -1: "anomaly"})
            st.session_state["features_df"] = features_df
            st.sidebar.success("Anomaly detection complete!")

            # Visualize results
            st.write("## Anomaly Detection - Overlapping Line Plot")
            plt.figure(figsize=(12, 6))

            for segment in st.session_state["segments"]:
                file_name = segment["file_name"]
                phase_id = segment["phase_id"]
                signal = segment["signal"]

                # Get anomaly status for this segment
                anomaly_label = st.session_state["features_df"][
                    (st.session_state["features_df"]["file_name"] == file_name) &
                    (st.session_state["features_df"]["phase_id"] == phase_id)
                ]["anomaly"].values[0]

                # Set color based on anomaly status
                color = "red" if anomaly_label == "anomaly" else "blue"

                # Plot the signal
                plt.plot(signal, color=color, alpha=0.7)

            # Add labels and legend
            plt.title("Overlapping Line Plot of Welding Phases (Red = Anomaly, Blue = Normal)")
            plt.xlabel("Time")
            plt.ylabel("Signal Amplitude")
            plt.grid(True)

            # Display the plot
            st.pyplot(plt)

if __name__ == "__main__":
    main()
