import os
import pickle

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
def print_mat_file(file_path):
    try:
        # Load the MATLAB file
        data = scipy.io.loadmat(file_path)

        # Print the contents of the MATLAB file
        print("MATLAB File Contents:")
        for variable_name in data:
            print(f"Variable: {variable_name}")
            print(data[variable_name])
            print("\n")
    except Exception as e:
        print(f"Error: {e}")

def plot_spikes(fluorescence_array, fps = 30):
    #interactive view
    number_consecutive_recordings=6
    # Calculate time values based on the frame rate (30 fps)
    fluorescence_array

    n = len(fluorescence_array)
    time_values = np.arange(n) / fps

    # Plot intensity against time
    plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
    plt.plot(time_values, fluorescence_array, linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Fluorescence Intensity')
    plt.title('Fluorescence Intensity vs. Time')
    plt.grid(True)

    # Show the plot
    # plt.show()

    return plt


def safePlotsAsPNG(output_directory):
    array_of_plots=[]
    with open(r"./data/temp/array_of_plots.pkl", 'rb') as file:
        array_of_plots = pickle.load(file)


    # Ensure the output directory exists; create it if it doesn't
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through the array_of_plots and save each image as a separate PNG file
    for i, image_data in enumerate(array_of_plots):
        file_path = os.path.join(output_directory, f"image_{i}.png")

        with open(file_path, "wb") as file:
            file.write(image_data)

        print(f"Saved image {i} to {file_path}")

