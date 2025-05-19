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

# Function to compute the gradient of the signal
def compute_gradient(signal_data):
    """
    Computes the gradient (first derivative) of the signal.
    """
    return np.gradient(signal_data)

# Function to separate welding phases using gradient-based detection
def separate_welding_phases_by_gradient(signal_data, gradient_threshold, window_size=50):
    if signal_data.empty:  # Handle empty signal data
        return []

    # Compute the gradient of the signal
    gradient = compute_gradient(signal_data)

    # Smooth the gradient using a rolling mean to reduce noise
    smoothed_gradient = pd.Series(gradient).rolling(window=window_size, center=True).mean()

    # Create a mask for regions where the gradient exceeds the threshold
    welding_mask = smoothed_gradient.abs() > gradient_threshold
    welding_phases = []

    # Identify start and end indices of welding phases
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

        # Slider for selecting gradient threshold
        gradient_threshold = st.sidebar.slider(
            "Set Gradient Threshold for Welding Phase Segmentation",
            min_value=0.0, max_value=10.0, value=0.5, step=0.1
        )

        # Window size for smoothing the gradient
        window_size = st.sidebar.slider(
            "Set Smoothing Window Size",
            min_value=1, max_value=100, value=50, step=1
        )

        if st.sidebar.button("Segment Welding Phases"):
            segments = []
            for file_path in file_paths:
                try:
                    df = pd.read_csv(file_path, header=None)
                    signal = df[0]
                    file_name = os.path.basename(file_path)

                    # Separate welding phases using gradient-based method
                    welding_phases = separate_welding_phases_by_gradient(
                        signal_data=signal, 
                        gradient_threshold=gradient_threshold, 
                        window_size=window_size
                    )
                    for i, phase in enumerate(welding_phases):
                        segments.append({
                            "file_name": file_name,
                            "phase_id": i + 1,
                            "signal": phase
                        })

                    # Plot the original signal, gradient, and detected welding phases
                    plt.figure(figsize=(12, 6))
                    plt.plot(signal, label="Original Signal", color="black", alpha=0.7)
                    gradient = compute_gradient(signal)
                    smoothed_gradient = pd.Series(gradient).rolling(window=window_size, center=True).mean()
                    plt.plot(smoothed_gradient, label="Smoothed Gradient", color="red", alpha=0.7)

                    # Highlight segmented welding phases
                    for i, phase in enumerate(welding_phases):
                        start_idx = phase.index[0]
                        end_idx = phase.index[-1]
                        plt.axvspan(start_idx, end_idx, color="blue" if i == 0 else "green", alpha=0.3, label=f"Phase {i + 1}")

                    plt.title(f"Signal Segmentation for {file_name}")
                    plt.xlabel("Time")
                    plt.ylabel("Signal Value / Gradient")
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt)

                except Exception as e:
                    st.error(f"Error processing file {file_path}: {e}")
                    continue

            st.session_state["segments"] = segments
            st.sidebar.success("Welding phases segmented successfully!")
            st.write("## Segmented Welding Phases")
            st.write(f"Total segmented phases: {len(segments)}")
            for segment in segments:
                st.write(f"File: {segment['file_name']}, Phase ID: {segment['phase_id']}, Length: {len(segment['signal'])}")

if __name__ == "__main__":
    main()
