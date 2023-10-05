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
    plt.show()


import numpy
import pandas as pd
import scipy.io

# function to load table variable from MAT-file from https://stackoverflow.com/questions/25853840/load-matlab-tables-in-python-using-scipy-io-loadmat
def loadtablefrommat(matfilename, tablevarname, columnnamesvarname):
    """
    read a struct-ified table variable (and column names) from a MAT-file
    and return pandas.DataFrame object.
    """

    # load file
    mat = scipy.io.loadmat(matfilename)

    # get table (struct) variable
    tvar = mat.get(tablevarname)
    data_desc = mat.get(columnnamesvarname)
    types = tvar.dtype
    fieldnames = types.names

    # extract data (from table struct)
    data = None
    for idx in range(len(fieldnames)):
        if fieldnames[idx] == 'data':
            data = tvar[0][0][idx]
            break;

    # get number of columns and rows
    numcols = data.shape[1]
    numrows = data[0, 0].shape[0]

    # and get column headers as a list (array)
    data_cols = []
    for idx in range(numcols):
        data_cols.append(data_desc[0, idx][0])

    # create dict out of original table
    table_dict = {}
    for colidx in range(numcols):
        rowvals = []
        for rowidx in range(numrows):
            rowval = data[0,colidx][rowidx][0]
            if type(rowval) == numpy.ndarray and rowval.size > 0:
                rowvals.append(rowval[0])
            else:
                rowvals.append(rowval)
        table_dict[data_cols[colidx]] = rowvals
    return pd.DataFrame(table_dict)