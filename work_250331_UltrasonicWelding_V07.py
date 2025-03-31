# -*- coding: utf-8 -*-
import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
import plotly.express as px

# Function to extract ZIP files
def extract_zip(uploaded_file, extract_to="extracted_data"):
    if os.path.exists(extract_to):
        for file in os.listdir(extract_to):
            os.remove(os.path.join(extract_to, file))
    else:
        os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    return [os.path.join(extract_to, f) for f in os.listdir(extract_to) if f.endswith('.csv')]

# Function to compute rolling variance
def compute_rolling_variance(signal_data, window_size=50):
    return signal_data.rolling(window=window_size, center=True).var()

# Function to separate welding phases using variance threshold
def separate_welding_phases_by_variance(signal_data, threshold, window_size=50):
    if signal_data.empty:
        return []

    rolling_variance = compute_rolling_variance(signal_data, window_size=window_size)
    welding_mask = rolling_variance > threshold
    welding_phases = []

    change_points = np.where(np.diff(welding_mask.astype(int)) != 0)[0] + 1
    if welding_mask.iloc[0]:
        change_points = np.insert(change_points, 0, 0)
    if welding_mask.iloc[-1]:
        change_points = np.append(change_points, len(signal_data))

    for i in range(0, len(change_points) - 1, 2):
        phase = signal_data.iloc[change_points[i]:change_points[i + 1]]
        if len(phase) > 5000:
            if len(welding_phases) < 2:
                welding_phases.append(phase)
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

        variance_threshold = st.sidebar.slider(
            "Set Variance Threshold for Welding Phase Segmentation",
            min_value=0.0, max_value=0.1, value=0.01, step=0.001
        )

        if st.sidebar.button("Segment Welding Phases"):
            segments = []
            for file_path in file_paths:
                try:
                    df = pd.read_csv(file_path, header=None)
                    signal = df[0]
                    file_name = os.path.basename(file_path)
                    welding_phases = separate_welding_phases_by_variance(signal, threshold=variance_threshold)
                    for i, phase in enumerate(welding_phases):
                        segments.append({
                            "file_name": file_name,
                            "phase_id": i + 1,
                            "signal": phase,
                            "length": len(phase)  # Add length of the phase
                        })

                    plt.figure(figsize=(24, 3))
                    plt.plot(signal, label="Original Signal", color="black", alpha=0.7)
                    rolling_variance = compute_rolling_variance(signal)
                    plt.plot(rolling_variance, label="Rolling Variance", color="red", alpha=0.7)

                    for i, phase in enumerate(welding_phases):
                        start_idx = phase.index[0]
                        end_idx = phase.index[-1]
                        plt.axvspan(start_idx, end_idx, color="blue" if i == 0 else "green", alpha=0.3, label=f"Phase {i + 1}")

                    plt.title(f"Signal Segmentation for {file_name}")
                    plt.xlabel("Time")
                    plt.ylabel("Signal Value / Variance")
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
                st.write(f"File: {segment['file_name']}, Phase ID: {segment['phase_id']}, Length: {segment['length']}")

        if "segments" in st.session_state and st.sidebar.button("Extract Features"):
            segments = st.session_state["segments"]
            feature_list = []
            for segment in segments:
                features = extract_features(segment["signal"])
                feature_list.append([segment["file_name"], segment["phase_id"], segment["length"]] + features)

            feature_columns = ["file_name", "phase_id", "length"] + [f"feature_{i}" for i in range(10)]
            features_df = pd.DataFrame(feature_list, columns=feature_columns)
            st.session_state["features_df"] = features_df
            st.write("## Extracted Features")
            st.dataframe(features_df)

        # Input for number of clusters
        num_clusters = st.sidebar.number_input(
            "Number of Clusters for K-Means",
            min_value=2,
            max_value=10,
            value=3,
            step=1
        )

        if "features_df" in st.session_state and st.sidebar.button("Perform K-Means Clustering"):
            features_df = st.session_state["features_df"]
            selected_features = features_df.iloc[:, 3:]

            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(selected_features)
            features_df["cluster"] = clusters
            st.session_state["features_df"] = features_df
            st.sidebar.success("Clustering complete!")

            # Perform PCA for 3D visualization
            pca = PCA(n_components=3)
            pca_values = pca.fit_transform(selected_features)

            # Create DataFrame for Plotly Plot
            plot_df = features_df.copy()
            plot_df["pca_1"] = pca_values[:, 0]
            plot_df["pca_2"] = pca_values[:, 1]
            plot_df["pca_3"] = pca_values[:, 2]

            # Plotly 3D Scatter Plot
            fig = px.scatter_3d(
                plot_df,
                x="pca_1",
                y="pca_2",
                z="pca_3",
                color=plot_df["cluster"].astype(str),  # Use distinct colors for clusters
                symbol="phase_id",  # Different symbols for Phase ID
                hover_data={
                    "file_name": True,
                    "phase_id": True,
                    "length": True,
                    "cluster": True,
                },
                title="3D PCA Visualization of Clusters",
            )

            fig.update_traces(marker=dict(size=8, opacity=0.7))
            fig.update_layout(
                height=800,  # Adjust height to match the main section
                scene=dict(aspectmode="cube"),  # Keep aspect ratio consistent
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
