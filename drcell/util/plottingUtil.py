import io
import os

import numpy as np
from matplotlib import pyplot as plt


def plot_and_return_spikes(trace_data_arrays, indices, fps=30, number_consecutive_recordings=1,
                           background_traces=False, recording_type=None):
    """Plot spike times: if one array in fluorescence_arrays, then raw data is plotted if multiple arrays in fluorescence_arrays, then median gets plotted.

    Parameters
    ----------
    trace_data_arrays : np.ndarray
        2-dimensional array with trace data
    indices : list
        list of indices of traces to be plotted from the fluorescence_arrays
    fps : int
        frames per second of recording (to calculate the time values of the samples)
    number_consecutive_recordings : int
        number of recordings that are stitched together (relevant for dividing lines in plot)
    background_traces : boolean
        if True plots the raw traces of the selected indices gray in the background (applies only with multiple traces)

    Returns
    -------
    matplotlib.pyplot
        plot of spike trace


    """

    # Calculate time values based on the frame rate per second
    n = trace_data_arrays.shape[1]
    time_values = np.arange(n) / fps

    # Plot intensity against time
    plt.figure(figsize=(10, 4))  # Adjust the figure size as needed

    # takes just the arrays with corresponding indices
    selected_arrays = trace_data_arrays[indices]
    # makes median over all
    median_selected_arrays = np.median(selected_arrays, axis=0)

    plt.xlabel('Time (s)')

    if recording_type == "Ephys":
        plt.ylabel('Firing rate (spikes/s)')
        plt.title('Firing rate vs. Time')
        if background_traces:
            for selected_fluorescence_array in selected_arrays:
                plt.plot(time_values, selected_fluorescence_array, linestyle='-', color='gray', alpha=0.3)

        plt.plot(time_values, median_selected_arrays, linestyle='-')
    elif recording_type == "2P":
        plt.ylabel('Fluorescence Intensity')
        plt.title('Fluorescence Intensity vs. Time')
        if background_traces:
            for selected_fluorescence_array in selected_arrays:
                plt.plot(time_values, selected_fluorescence_array, linestyle='-', color='gray', alpha=0.3)

        plt.plot(time_values, median_selected_arrays, linestyle='-')
    else:
        plt.ylabel('Dimension y-Axis')
        plt.title('Original data')
        if background_traces:
            for selected_fluorescence_array in selected_arrays:
                plt.plot(range(n), selected_fluorescence_array, linestyle='-', color='gray', alpha=0.3)

        plt.plot(range(n), median_selected_arrays, linestyle='-')
    plt.grid(True)

    # Add vertical lines at specific x-coordinates (assuming they have the same recording length)
    for i in range(1, number_consecutive_recordings):
        plt.axvline(x=((trace_data_arrays.shape[1] / fps) / number_consecutive_recordings) * i, color='black',
                    linestyle='--')

    # Show the plot
    # plt.show()

    return plt


def plot_and_return_pca_plot(data_arrays, indices, background_traces=False):
    """Plot spike times: if one array in fluorescence_arrays, then raw data is plotted if multiple arrays in fluorescence_arrays, then median gets plotted.

    Parameters
    ----------
    data_arrays : np.ndarray
        2-dimensional array with trace data
    indices : list
        list of indices of traces to be plotted from the pca preprocessed fluorescence_arrays
    background_traces : boolean
        if True plots the raw traces of the selected indices gray in the background (applies only with multiple traces)

    Returns
    -------
    matplotlib.pyplot
        plot of pca plot


    """

    # Calculate time values based on the frame rate per second
    n = data_arrays.shape[1]
    pca_components = np.arange(n)

    # Plot intensity against time
    plt.figure(figsize=(10, 4))  # Adjust the figure size as needed

    # takes just the arrays with corresponding indices
    selected_arrays = data_arrays[indices]
    # makes median over all

    median_selected_arrays = np.median(selected_arrays, axis=0)

    if background_traces:
        for selected_fluorescence_array in selected_arrays:
            plt.plot(pca_components, selected_fluorescence_array, linestyle='-', color='gray', alpha=0.3)

    plt.plot(pca_components, median_selected_arrays, linestyle='-', color='red')
    plt.xlabel('PCA Component')
    plt.ylabel('Value')
    plt.title('Plot of Values of PCA Components')
    plt.grid(True)

    # Show the plot
    # plt.show()

    return plt


def get_plot_for_indices(trace_arrays, indices, fps=30, number_consecutive_recordings=1, extend_plot=False,
                         recording_type=None):
    if trace_arrays is None or indices is None:
        print("No image to plot")
        # Create a figure with a black background
        fig = plt.figure(facecolor='black')
        ax = fig.add_subplot(111, facecolor='black')

        # Hide axis and grid lines
        ax.axis('off')
        return plt
    else:
        return plot_and_return_spikes(trace_arrays, indices, fps=fps,
                                      number_consecutive_recordings=number_consecutive_recordings,
                                      background_traces=extend_plot, recording_type=recording_type)


def get_pca_plot_for_indices(trace_arrays, indices, extend_plot=False):
    if trace_arrays is None or indices is None:
        print("No image to plot")
        # Create a figure with a black background
        fig = plt.figure(facecolor='black')
        ax = fig.add_subplot(111, facecolor='black')

        # Hide axis and grid lines
        ax.axis('off')
        return plt
    else:
        return plot_and_return_pca_plot(trace_arrays, indices, background_traces=extend_plot)


def plot_and_save_spikes(neuron_number, dataframe, output_folder, fps=30, number_consecutive_recordings=6):
    # takes selected row (fluorescence data of one cell), makes it to an array and plots it
    plt = plot_and_return_spikes(dataframe.values, neuron_number, fps=fps,
                                 number_consecutive_recordings=number_consecutive_recordings)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80)
    buf.seek(0)
    # Ensure the output directory exists; create it if it doesn't
    os.makedirs(output_folder, exist_ok=True)

    # Save each image as a separate PNG file
    file_path = os.path.join(output_folder, f"image_{neuron_number}.png")
    with open(file_path, "wb") as file:
        file.write(buf.read())

    print(f"Saved image {neuron_number} to {file_path}")
    plt.close()
