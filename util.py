import json
import os
import pickle
import socket
from decimal import Decimal

import h5py
import numpy as np
import pandas as pd
import phate
import scipy.io as sio
import sklearn.decomposition  # PCA
import umap

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


def get_dimensional_reduction_out(reduction_function_name, data, dump_folder_path, reduction_functions,
                                  reduction_params,
                                  pca_preprocessing=False, pca_n_components=2, show_pca_diagnostic_plot=False,
                                  output_buffer_param_dump_filename_extension="_parameter_buffer_dump.pkl"):
    dataset_name = os.path.basename(dump_folder_path)
    dump_folder_path = os.path.join(dump_folder_path,
                                    f"{reduction_function_name}" + output_buffer_param_dump_filename_extension)

    if reduction_function_name == "None" or reduction_function_name is None:
        pca_preprocessing = True
        pca_n_components = 2
        param_key = (("pca_preprocessing_n_components", pca_n_components),)
        reduction_params = {"pca_preprocessing_n_components": pca_n_components}
    else:
        # sort parameters alphabetically, to prevent duplicates in dump file
        reduction_params_items = list(reduction_params.items())
        reduction_params_items.sort()
        if pca_preprocessing:
            param_key = tuple(reduction_params_items) + (("pca_preprocessing_n_components", pca_n_components),)
        else:
            param_key = tuple(reduction_params_items)
    buffered_data_dump = {}

    # Checks if the dump file with this path was already called.
    # If so, instead of loading it for every call of the function, it takes the data from there
    if dump_folder_path not in dimensional_reduction_out_dump_data:
        # Check if the file exists
        if os.path.exists(dump_folder_path):
            with open(dump_folder_path, 'rb') as file:
                dimensional_reduction_out_dump_data[dump_folder_path] = pickle.load(file)
        else:
            # If the file doesn't exist, create it and write something to it
            with open(dump_folder_path, 'wb') as file:
                pickle.dump(buffered_data_dump, file)
                dimensional_reduction_out_dump_data[dump_folder_path] = buffered_data_dump

            print(f"The file '{dump_folder_path}' has been created.")

    buffered_data_dump = dimensional_reduction_out_dump_data[dump_folder_path]
    current_data = data

    if param_key not in buffered_data_dump:
        if pca_preprocessing:
            print(
                f"Generate {reduction_function_name} with PCA preprocessing: File = {dataset_name}/{os.path.basename(dump_folder_path)}, {reduction_params}, PCA n_components = {pca_n_components}")
            current_data = apply_pca_preprocessing(current_data, n_components=pca_n_components,
                                                   show_diagnostic_plot=show_pca_diagnostic_plot)
        else:
            print(
                f"Generate {reduction_function_name}: File = {dataset_name}/{os.path.basename(dump_folder_path)}, {reduction_params}")

        if reduction_function_name == "None" or reduction_function_name == None:
            reduced_data = current_data
        else:
            reduced_data = reduction_functions[reduction_function_name]["function"](current_data, **reduction_params)

        buffered_data_dump[param_key] = reduced_data

        with open(dump_folder_path, 'wb') as file:
            dimensional_reduction_out_dump_data[dump_folder_path] = buffered_data_dump
            pickle.dump(buffered_data_dump, file)

    print(
        f"Return {reduction_function_name}: File = {dataset_name}/{os.path.basename(dump_folder_path)}, {reduction_params}")
    return buffered_data_dump[param_key]


def load_and_preprocess_mat_data_AD_IL(c_file, recording_type='2P'):
    # load data
    g = sio.loadmat(c_file)
    # g = h5py.File(cFile,'r')
    # X = np.array(g['X'])
    X = np.array(g['UMAPmatrix'])
    Y = np.array(g['UMAPmatrix'])
    # Y = np.array(g['Y'])
    # Y = np.transpose(Y)

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
    if recording_type == '2P':
        # Extract the 'matrixLegendArray' array from the loaded data and create a Pandas DataFrame from 'matrixLegendArray'
        matrix_legend_df = pd.DataFrame(g['matrixLegendArray'],
                                        columns=["Animal", "Recording", "Neuron", "NbFrames", "Task", "RedNeurons",
                                                 "ChoiceAUCs", "IsChoiceSelect", "StimAUCs", "IsStimSelect"])

        # Convert the 'Task' column to integers
        matrix_legend_df['Task'] = (matrix_legend_df['Task'].astype(int)).astype(str)
        # Convert Float to Boolean
        matrix_legend_df["IsStimSelect"] = matrix_legend_df["IsStimSelect"].apply(lambda x: x >= 1.0)
        matrix_legend_df["IsChoiceSelect"] = matrix_legend_df["IsChoiceSelect"].apply(lambda x: x >= 1.0)
        matrix_legend_df["RedNeurons"] = matrix_legend_df["RedNeurons"].apply(lambda x: x >= 1.0)
    elif recording_type == 'Ephys':
        # Extract the 'matrixLegendArray' array from the loaded data and create a Pandas DataFrame from 'matrixLegendArray'
        matrix_legend_df = pd.DataFrame(g['matrixLegend'],
                                        columns=["Area", "Neuron", "ChoiceAUCsVisual", "ChoiceAUCsTactile",
                                                 "ChoiceAUCsMultisensory", "IsChoiceSel", "IsChoiceSelVisual",
                                                 "IsChoiceSelTactile", "IsChoiceSelMultisensory", "StimAUCsVisual",
                                                 "StimAUCsTactile", "StimAUCsMultisensory", "IsStimSel",
                                                 "IsStimSelVisual", "IsStimSelTactile", "IsStimSelMultisensory"])

        # Convert the 'Task' column to integers
        matrix_legend_df['Area'] = matrix_legend_df['Area'].astype(int)
        brain_area_mapping = ["V1", "Superficial SC", "Deep SC", "ALM", "Between ALM and MM", "MM"]
        matrix_legend_df['Area'] = matrix_legend_df['Area'].apply(lambda x: f'{brain_area_mapping[x - 1]}')

        # Convert Float to Boolean
        matrix_legend_df["IsStimSel"] = matrix_legend_df["IsStimSel"].apply(lambda x: x >= 1.0)
        matrix_legend_df["IsChoiceSel"] = matrix_legend_df["IsChoiceSel"].apply(lambda x: x >= 1.0)
        matrix_legend_df["IsChoiceSelVisual"] = matrix_legend_df["IsChoiceSelVisual"].apply(lambda x: x >= 1.0)
        matrix_legend_df["IsChoiceSelTactile"] = matrix_legend_df["IsChoiceSelTactile"].apply(lambda x: x >= 1.0)
        matrix_legend_df["IsChoiceSelMultisensory"] = matrix_legend_df["IsChoiceSelMultisensory"].apply(
            lambda x: x >= 1.0)
        matrix_legend_df["IsStimSel"] = matrix_legend_df["IsStimSel"].apply(lambda x: x >= 1.0)
        matrix_legend_df["IsStimSelVisual"] = matrix_legend_df["IsStimSelVisual"].apply(lambda x: x >= 1.0)
        matrix_legend_df["IsStimSelTactile"] = matrix_legend_df["IsStimSelTactile"].apply(lambda x: x >= 1.0)
        matrix_legend_df["IsStimSelMultisensory"] = matrix_legend_df["IsStimSelMultisensory"].apply(lambda x: x >= 1.0)

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


def convert_data_AD_IL(input_file_path, output_path, recording_type=None):
    data_variables = []
    display_hover_variables = []
    if recording_type == "Ephys":
        # variables from the input data, that is selectable in the Color and Filter setting
        data_variables = ["IsChoiceSel", "IsStimSel", "Area", "IsStimSelVisual", "IsStimSelTactile",
                          "IsStimSelMultisensory",
                          "IsChoiceSelVisual", "IsChoiceSelTactile", "IsChoiceSelMultisensory"]
        # variables from the input data, that gets displayed in the hover tool
        display_hover_variables = ["pdIndex", "Neuron", "Area", "ChoiceAUCsVisual", "ChoiceAUCsTactile",
                                   "ChoiceAUCsMultisensory", "StimAUCsVisual", "StimAUCsTactile",
                                   "StimAUCsMultisensory"]
    elif recording_type == "2P":
        # variables from the input data, that is selectable in the Color and Filter setting
        data_variables = ["IsChoiceSelect", "IsStimSelect", "Task", "RedNeurons"]
        # variables from the input data, that gets displayed in the hover tool
        display_hover_variables = ["pdIndex", "Neuron", "ChoiceAUCs", "StimAUCs"]

    titles = ["all", "excludeChoiceUnselectBefore", "excludeStimUnselectBefore"]
    umap_df, matrix_legend_df, cleaned_data = load_and_preprocess_mat_data_AD_IL(input_file_path,
                                                                                 recording_type=recording_type)
    recording_types = {}
    cleaned_data_dfs = {}
    matrix_legend_dfs = {}
    output_files = {}
    # filter out specific values beforehand and add them as seperate datasets
    for title in titles:
        if recording_type is None or recording_type == "None":
            config = None
        else:
            config = {
                "recording_type": recording_type,
                "data_variables": data_variables,
                "display_hover_variables": display_hover_variables,
            }
        cleaned_data_dfs[title] = pd.DataFrame(cleaned_data)
        matrix_legend_dfs[title] = matrix_legend_df
        recording_types[title] = recording_type
        if title == "all":
            print(f"{title} Data Length: {len(matrix_legend_df)}")

        elif title == "excludeChoiceUnselectBefore":
            # Filters cells with Property
            cleaned_data_dfs[title] = pd.DataFrame(cleaned_data[matrix_legend_df[data_variables[0]]])
            matrix_legend_dfs[title] = matrix_legend_df[matrix_legend_df[data_variables[0]]]
            print(f"{title} Data Length: {matrix_legend_df[data_variables[0]].apply(lambda x: x).sum()}")

        elif title == "excludeStimUnselectBefore":

            # Filters cells with Property
            cleaned_data_dfs[title] = pd.DataFrame(cleaned_data[matrix_legend_df[data_variables[1]]])
            matrix_legend_dfs[title] = matrix_legend_df[matrix_legend_df[data_variables[1]]]
            print(f"{title} Data Length: {matrix_legend_df[data_variables[1]].apply(lambda x: x).sum()}")

        output_files[title] = os.path.join(output_path,
                                           os.path.splitext(os.path.basename(input_file_path))[0] + f"_{title}.h5")
        save_as_dr_cell_h5(output_files[title], cleaned_data_dfs[title], matrix_legend_dfs[title], config=config)
    return list(output_files.values())


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


def generate_umap(data, n_neighbors, min_dist, n_components=2, random_state=42):
    umap_object = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    return umap_object.fit_transform(data)


def generate_umap_diagnostic_plot(data, n_neighbors, min_dist, n_components=2, random_state=42):
    umap_object = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    return plotting.generate_diagnostic_plots(umap_object, data)


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


def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0


# TODO maybe change to easier hdf5 structure for direct matlab export
def save_as_dr_cell_h5(file, data_df, legend_df, config=None):
    # Save dataframes and config to HDF5
    with h5py.File(file, 'w') as hdf:

        # first version
        # # Convert DataFrames to numpy arrays and process them
        # data_df_compatible = np.array(
        #     [convert_to_hdf5_compatible(data_df[col].to_numpy()) for col in data_df.columns]).T
        # legend_df_compatible = np.array(
        #     [convert_to_hdf5_compatible(legend_df[col].to_numpy()) for col in legend_df.columns]).T
        #
        # # Save dataframes as datasets
        # hdf.create_dataset('data_df', data=data_df_compatible)
        # hdf.create_dataset('legend_df', data=legend_df_compatible)
        #
        # # Add dataframe columns as attributes
        # hdf['data_df'].attrs['columns'] = data_df.columns.tolist()
        # hdf['legend_df'].attrs['columns'] = legend_df.columns.tolist()

        if config is not None:
            # Save config as attributes
            for key, value in config.items():
                hdf.attrs[key] = value

    with pd.HDFStore(file) as store:
        store.append('data_df', data_df)
        store.append('legend_df', legend_df)


def load_dr_cell_h5(file):
    with h5py.File(file, 'r') as hdf:
        # first version
        # # Read dataframes
        # data_df = pd.DataFrame(hdf['data_df'][:], columns=hdf['data_df'].attrs['columns'])
        # legend_df = pd.DataFrame(hdf['legend_df'][:], columns=hdf['legend_df'].attrs['columns'])

        # Read config from attributes
        config = {key: hdf.attrs[key] for key in hdf.attrs}
        if config is None:
            with open('default_file_config.json', 'r') as json_file:
                config = json.load(json_file)

    # Read DataFrames from the HDF5 file
    with pd.HDFStore(file) as store:
        data_df = store['data_df']
        legend_df = store['legend_df']
    return data_df, legend_df, config


def convert_to_hdf5_compatible(array):
    # Convert object dtypes to fixed-length byte strings
    if array.dtype == object:
        return array.astype(str).astype('S')
    return array


def create_file_folder_structure(output_path, folder_name):
    # path created for the data of the given input file
    file_folder_path = os.path.join(output_path, folder_name)
    # folder for output file for exports of corresponding input file
    file_folder_output_path = os.path.join(file_folder_path, "output")

    print(f"Using '{folder_name}' as input.")
    # creates a folder for the corresponding input file, where data gets saved
    if not os.path.exists(file_folder_path):
        # If not, create it
        os.makedirs(file_folder_path)
        os.makedirs(file_folder_output_path)
        print(f"Folder '{file_folder_path}' created.")
    else:
        if not os.path.exists(file_folder_output_path):
            os.makedirs(file_folder_output_path)

        print(f"Folder '{file_folder_path}' already exists.")

    return file_folder_path
