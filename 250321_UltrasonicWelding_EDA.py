# -*- coding: utf-8 -*-
import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to extract ZIP files
def extract_zip(uploaded_file, extract_to="extracted_data"):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return [os.path.join(extract_to, f) for f in os.listdir(extract_to) if f.endswith('.csv')]

# Function to segment welding phases based on amplitude threshold
def segment_welding_phases(signal_data, amplitude_threshold):
    """
    Segments the signal into welding phases based on amplitude thresholds.
    """
    # Compute absolute values of the signal (for symmetry around 0)
    abs_signal = np.abs(signal_data)

    # Create a mask for values above the threshold
    welding_mask = abs_signal > amplitude_threshold
    welding_phases = []

    # Identify start and end indices of welding phases
    change_points = np.where(np.diff(welding_mask.astype(int)) != 0)[0] + 1
    if welding_mask.iloc[0]:  # If signal starts in a welding phase
        change_points = np.insert(change_points, 0, 0)
    if welding_mask.iloc[-1]:  # If signal ends in a welding phase
        change_points = np.append(change_points, len(signal_data))

    # Extract segments corresponding to welding phases
    for i in range(0, len(change_points) - 1, 2):
        welding_phases.append(signal_data.iloc[change_points[i]:change_points[i + 1]])

    return welding_phases

# Main function
def main():
    st.sidebar.title("Ultrasonic Welding Data Explorer")
    uploaded_file = st.sidebar.file_uploader("Upload ZIP file containing CSVs", type=["zip"])

    if uploaded_file:
        file_paths = extract_zip(uploaded_file)
        st.sidebar.write(f"Total CSV files: {len(file_paths)}")

        # Slider for selecting amplitude threshold
        amplitude_threshold = st.sidebar.slider(
            "Set Amplitude Threshold for Welding Phase Segmentation",
            min_value=0.0, max_value=10.0, value=1.0, step=0.1
        )

        if st.sidebar.button("Segment Welding Phases"):
            segments = []
            for file_path in file_paths:
                try:
                    df = pd.read_csv(file_path, header=None)
                    signal = df[0]
                    file_name = os.path.basename(file_path)

                    # Segment welding phases based on amplitude
                    welding_phases = segment_welding_phases(signal_data=signal, amplitude_threshold=amplitude_threshold)
                    for i, phase in enumerate(welding_phases):
                        segments.append({
                            "file_name": file_name,
                            "phase_id": i + 1,
                            "signal": phase
                        })

                    # Plot the original signal and highlight welding phases
                    plt.figure(figsize=(12, 6))
                    plt.plot(signal, label="Original Signal", color="black", alpha=0.7)

                    # Highlight segmented welding phases
                    for i, phase in enumerate(welding_phases):
                        start_idx = phase.index[0]
                        end_idx = phase.index[-1]
                        plt.axvspan(start_idx, end_idx, color="blue" if i == 0 else "green", alpha=0.3, label=f"Phase {i + 1}")

                    plt.title(f"Signal Segmentation for {file_name}")
                    plt.xlabel("Time")
                    plt.ylabel("Signal Amplitude")
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
