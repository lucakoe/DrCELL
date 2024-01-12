import os
import pickle
import sklearn.decomposition  # PCA
import pandas as pd
import scipy.io as sio
import numpy as np
import umap
import matplotlib.pyplot as plt
import phate
from decimal import Decimal

import plotting

dimensional_reduction_out_dump_data = {}


def print_mat_file(file_path):
    try:
        # Load the MATLAB file
        data = sio.loadmat(file_path)

        # Print the contents of the MATLAB file
        print("MATLAB File Contents:")
        for variable_name in data:
            print(f"Variable: {variable_name}")
            print(data[variable_name])
            print("\n")
    except Exception as e:
        print(f"Error: {e}")


def get_decimal_places(number):
    return len(str(number).split('.')[1])

    decimal_places = Decimal(str(number)).as_tuple().exponent
    return max(0, -decimal_places)


def apply_pca_preprocessing(data, n_components=2, show_diagnostic_plot=False):
    """
    Apply PCA preprocessing to the input data.

    Parameters:
    - data: The input data.
    - n_components: The number of components for PCA.
    - show_diagnostic_plot: Whether to show the PCA diagnostic plot.

    Returns:
    - The PCA preprocessed data.
    """
    pca_operator = sklearn.decomposition.PCA(n_components=n_components)
    pca_data = pca_operator.fit_transform(data)

    if show_diagnostic_plot:
        diagnostic_data = pca_operator.explained_variance_ratio_
        diagnostic_plot = plotting.return_pca_diagnostic_plot(diagnostic_data)
        diagnostic_plot.show()

    return pca_data


def get_dimensional_reduction_out(reduction_function_name, data, dump_path, reduction_functions, reduction_params,
                                  pca_preprocessing=False, pca_n_components=2, show_pca_diagnostic_plot=False):
    dump_path = os.path.join(os.path.dirname(dump_path), f"{reduction_function_name}_" + os.path.basename(dump_path))

    if pca_preprocessing:
        param_key = tuple(reduction_params.items()) + (("pca_preprocessing_n_components", pca_n_components),)
    else:
        param_key = tuple(reduction_params.items())

    buffered_data_dump = {}

    # Checks if the dump file with this path was already called.
    # If so, instead of loading it for every call of the function, it takes the data from there
    if dump_path not in dimensional_reduction_out_dump_data:
        # Check if the file exists
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as file:
                dimensional_reduction_out_dump_data[dump_path] = pickle.load(file)
        else:
            # If the file doesn't exist, create it and write something to it
            with open(dump_path, 'wb') as file:
                pickle.dump(buffered_data_dump, file)
                dimensional_reduction_out_dump_data[dump_path] = buffered_data_dump

            print(f"The file '{dump_path}' has been created.")

    buffered_data_dump = dimensional_reduction_out_dump_data[dump_path]
    current_data = data

    if param_key not in buffered_data_dump:
        if pca_preprocessing:
            print(
                f"Generate {reduction_function_name} with PCA preprocessing: File = {os.path.basename(dump_path)}, {reduction_params}, PCA n_components = {pca_n_components}")
            current_data = apply_pca_preprocessing(current_data, n_components=pca_n_components,
                                                   show_diagnostic_plot=show_pca_diagnostic_plot)

        print(f"Generate {reduction_function_name}: File = {os.path.basename(dump_path)}, {reduction_params}")
        reduced_data = reduction_functions[reduction_function_name]["function"](current_data, **reduction_params)

        buffered_data_dump[param_key] = reduced_data

        with open(dump_path, 'wb') as file:
            dimensional_reduction_out_dump_data[dump_path] = buffered_data_dump
            pickle.dump(buffered_data_dump, file)

    print(f"Return {reduction_function_name}: File = {os.path.basename(dump_path)}, {reduction_params}")
    return buffered_data_dump[param_key]


# depreciated
def get_umap_out(data, dump_path, n_neighbors=20,
                 min_dist=0.0,
                 n_components=2,
                 random_state=42, pca_preprocessing=False, pca_n_components=2, show_pca_diagnostic_plot=False):
    if pca_preprocessing:
        param_key = (n_neighbors, min_dist, n_components, random_state, pca_n_components)
    else:
        param_key = (n_neighbors, min_dist, n_components, random_state)
    umap_out_df_dump = {}

    # Checks the global variable if the dumpfile with this path was already called.
    # If so instead of loading it for every call of the function it takes the data from there
    if not (dump_path in dimensional_reduction_out_dump_data):
        # Check if the file exists
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as file:
                dimensional_reduction_out_dump_data[dump_path] = pickle.load(file)
        else:
            # If the file doesn't exist, create it and write something to it
            with open(dump_path, 'wb') as file:
                pickle.dump(umap_out_df_dump, file)
                dimensional_reduction_out_dump_data[dump_path] = umap_out_df_dump

            print(f"The file '{dump_path}' has been created.")

    umap_out_df_dump = dimensional_reduction_out_dump_data[dump_path]
    current_data = data
    if not (param_key in umap_out_df_dump):
        if pca_preprocessing:
            print(
                f"Generate UMAP with PCA preprocessing: File = {os.path.basename(dump_path)}, n_neighbors = {n_neighbors}, min_dist = {min_dist}, pca_n_components = {pca_n_components}")
            pca_operator = sklearn.decomposition.PCA(n_components=pca_n_components)
            pca = pca_operator.fit_transform(current_data)

            current_data = pca

            if show_pca_diagnostic_plot:
                diagnostic_data = pca_operator.explained_variance_ratio_
                diagnostic_plot = plotting.return_pca_diagnostic_plot(diagnostic_data)
                diagnostic_plot.show()

        else:
            print(
                f"Generate UMAP: File = {os.path.basename(dump_path)}, n_neighbors = {n_neighbors}, min_dist = {min_dist}")

        umap_object = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state,
        )
        umap_out_df_dump[param_key] = umap_object.fit_transform(current_data)

        with open(dump_path, 'wb') as file:
            dimensional_reduction_out_dump_data[dump_path] = umap_out_df_dump
            pickle.dump(umap_out_df_dump, file)
    if pca_preprocessing:
        print(
            f"Return UMAP with PCA preprocessing: File = {os.path.basename(dump_path)}, n_neighbors = {n_neighbors}, min_dist = {min_dist}, pca_n_components = {pca_n_components}")
    else:
        print(
            f"Return UMAP: File = {os.path.basename(dump_path)}, n_neighbors = {n_neighbors}, min_dist = {min_dist}")
    return umap_out_df_dump[param_key]


def load_and_preprocess_data(c_file):
    # load data
    g = sio.loadmat(c_file)
    # g = h5py.File(cFile,'r')
    # X = np.array(g['X'])
    X = np.array(g['UMAPmatrix'])
    Y = np.array(g['UMAPmatrix'])
    # Y = np.array(g['Y'])
    # Y = np.transpose(Y)
    matrix_legend = np.array(g['matrixLegendArray'])
    # groups = np.array(g['redIdxAll'])
    # groups = np.reshape(groups,X.shape[0])
    # groups = groups.astype(int)

    # groups = np.array(g['groupIdx'])

    # groups = np.array(g['matrix_legend'])
    # for iNeuron in range(0,len(groups)):
    #     if groups[iNeuron,4] == 'Innate':
    #         groups[iNeuron,4] = 1
    #     elif groups[iNeuron,4] == 'Audio Task Without Delay':
    #         groups[iNeuron,4] = 2
    #     elif groups[iNeuron,4] == 'Audio Task with delay':
    #         groups[iNeuron,4] = 3
    #     elif groups[iNeuron,4] == 'Innate2':
    #         groups[iNeuron,4] = 4
    # groups = np.reshape(groups[:,4],X.shape[0])
    # groups = groups.astype(int)

    # g.close

    # Extract the 'UMAPmatrix' array from the loaded data and create a Pandas DataFrame from 'UMAPmatrix'
    umap_df = pd.DataFrame(g['UMAPmatrix'])

    # Extract the 'matrixLegendArray' array from the loaded data and create a Pandas DataFrame from 'matrixLegendArray'
    matrix_legend_df = pd.DataFrame(g['matrixLegendArray'],
                                    columns=["Animal", "Recording", "Neuron", "NbFrames", "Task", "RedNeurons",
                                             "ChoiceAUCs", "IsChoiceSelect", "StimAUCs", "IsStimSelect"])

    # Convert the 'Task' column to integers
    matrix_legend_df['Task'] = matrix_legend_df['Task'].astype(int)
    # Convert Float to Boolean
    matrix_legend_df["IsStimSelect"] = matrix_legend_df["IsStimSelect"].apply(lambda x: x >= 1.0)
    matrix_legend_df["IsChoiceSelect"] = matrix_legend_df["IsChoiceSelect"].apply(lambda x: x >= 1.0)
    matrix_legend_df["RedNeurons"] = matrix_legend_df["RedNeurons"].apply(lambda x: x >= 1.0)

    use_idx = np.invert(np.isnan(np.sum(Y, axis=0)))
    Y = Y[:, use_idx]

    # run umap twice, once for spatial and once for temporal clusters
    # umapOut = umap.UMAP(
    #     n_neighbors=30,
    #     min_dist=0.0,
    #     n_components=2,
    #     random_state=42,
    #     ).fit_transform(X)
    # print('Umap vals: ' + str(umapOut.shape))

    # show results
    # cIdx = np.random.permutation(groups.size)
    # fig1, ax1 = plt.subplots()
    # cColor = sns.color_palette("Dark2", np.max([groups])+1);
    # ax1.scatter(umapOut[cIdx, 0], umapOut[cIdx, 1], c=[cColor[x] for x in groups[cIdx]], s=2)
    # #plt.axis('square')
    # ax1.set_aspect('equal', adjustable = 'datalim')
    # plt.show()

    # temporal clusters
    return umap_df, matrix_legend_df, Y


import colorsys


def generate_color_palette(num_colors):
    """
    Generate a list of distinct colors based on the number of colors needed.

    Args:
        num_colors (int): The number of colors to generate.

    Returns:
        list: A list of distinct colors in hexadecimal format, e.g., ['#FF0000', '#00FF00', ...]
    """

    def rgb_to_hex(rgb):
        """
        Convert an RGB color tuple to a hexadecimal color string.

        Args:
            rgb (tuple): An RGB color tuple, e.g., (255, 0, 0).

        Returns:
            str: Hexadecimal color string, e.g., '#FF0000'.
        """
        # Ensure the RGB values are integers
        r, g, b = [int(x) for x in rgb]
        return '#{:02X}{:02X}{:02X}'.format(r, g, b)

    if num_colors <= 0:
        return []

    # Create a list of evenly spaced hue values
    hue_values = [i / num_colors for i in range(num_colors)]

    # Convert hue to RGB colors
    colors = [rgb_to_hex(colorsys.hsv_to_rgb(hue, 1, 1)) for hue in hue_values]

    return colors


def generate_grid(min_point, max_point, center_point=(0.0, 0.0), grid_size_x=1, grid_size_y=1):
    # Define grid parameters
    center_x = center_point[0]  # Center x-coordinate of the grid
    center_y = center_point[1]  # Center y-coordinate of the grid
    min_x = min_point[0]  # Minimum x-coordinate
    max_x = max_point[0]  # Maximum x-coordinate
    min_y = min_point[1]  # Minimum y-coordinate
    max_y = max_point[1]  # Maximum y-coordinate

    # Calculate the number of grid lines in each direction
    num_x_lines_left = int((center_x - min_x) / grid_size_x)
    num_x_lines_right = int((max_x - center_x) / grid_size_x)
    num_y_lines_top = int((max_y - center_y) / grid_size_y)
    num_y_lines_bottom = int((center_y - min_y) / grid_size_y)

    # Generate data points for the grid and centers of squares
    grid_data = {'gridID': [], 'gridX': [], 'gridY': [], 'centerX': [], 'centerY': []}
    current_id = 0
    for i in range(-(num_x_lines_left + 1), num_x_lines_right + 1):
        for j in range(-(num_y_lines_bottom + 1), num_y_lines_top + 1):
            current_id += 1
            x = center_x + i * grid_size_x
            y = center_y + j * grid_size_y
            grid_data['gridID'].append(current_id)
            grid_data['gridX'].append(x)
            grid_data['gridY'].append(y)
            grid_data['centerX'].append(x + grid_size_x / 2)
            grid_data['centerY'].append(y + grid_size_y / 2)

    return pd.DataFrame(grid_data)


def assign_points_to_grid(points_df, grid_df, new_column_grid_df_name_and_property=[('index', 'pointIndices')]):
    # Initialize a new column in the grid DataFrame to store point indices
    for name_and_property in new_column_grid_df_name_and_property:
        grid_df[name_and_property[1]] = None

    for index, grid_row in grid_df.iterrows():
        x1, y1 = grid_row['gridX'], grid_row['gridY']
        x2, y2 = x1 + grid_row['gridSizeX'], y1 + grid_row['gridSizeY']

        # Find the points within the current grid cell
        points_in_grid = points_df[(points_df['x'] >= x1) & (points_df['x'] < x2) &
                                   (points_df['y'] >= y1) & (points_df['y'] < y2)]

        for name_and_property in new_column_grid_df_name_and_property:
            if name_and_property[0] == 'index':
                grid_df.at[index, name_and_property[1]] = points_in_grid.index
            else:
                grid_df.at[index, name_and_property[1]] = points_in_grid[name_and_property[0]]

    return grid_df


def generate_umap_parameters(data, dump_files_path, n_neighbors_values, min_dist_values, pca_n_components,
                             n_components_values=[2]):
    for n_neighbors_value in n_neighbors_values:
        for min_dist_value in min_dist_values:
            for n_components_value in n_components_values:
                get_umap_out(data,
                             os.path.abspath(dump_files_path),
                             n_neighbors=n_neighbors_value,
                             min_dist=min_dist_value, n_components=n_components_value)
                if n_components_value > 3:
                    for pca_n_components_value in pca_n_components:
                        get_umap_out(data,
                                     os.path.abspath(os.path.abspath(dump_files_path)).replace("\\",
                                                                                               '/'),
                                     n_neighbors=n_neighbors_value, min_dist=round(min_dist_value, 2),
                                     n_components=n_components_value,
                                     pca_n_components=int(pca_n_components_value),
                                     pca_preprocessing=True)


def generate_umap(data, n_neighbors, min_dist, n_components=2, random_state=42):
    umap_object = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    return umap_object.fit_transform(data)


def generate_t_sne(data, perplexity=30, learning_rate=200, n_iter=1000, early_exaggeration=12, angle=0.5,
                   metric="euclidean", n_components=2):
    tsne_operator = sklearn.manifold.TSNE(perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter,
                                          early_exaggeration=early_exaggeration, angle=angle,
                                          metric=metric, n_components=n_components)
    return tsne_operator.fit_transform(data)


def generate_phate(data, knn=30, decay=15, t='auto', gamma=0, n_jobs=-1, n_pca=100, n_landmark=1000, verbose=False,
                   n_components=2):
    phate_operator = phate.PHATE(n_jobs=n_jobs)
    phate_operator.set_params(knn=knn, decay=decay, t=t, gamma=gamma, n_jobs=n_jobs, n_pca=n_pca, n_landmark=n_landmark,
                              verbose=verbose, n_components=n_components)

    return phate_operator.fit_transform(data)
