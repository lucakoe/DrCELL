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
