import json
import os

import h5py
import numpy as np
import pandas as pd
from scipy import io as sio


# TODO fix weird behavior when converting data to h5 and back
def save_as_dr_cell_h5(file: str, data_df: pd.DataFrame, legend_df: pd.DataFrame, config: dict = None) -> None:
    # Save dataframes and config to HDF5
    with h5py.File(file, 'w') as hdf:
        # Create HDF5 file and store DataFrames
        save_dataframe_to_hdf5(hdf, data_df, 'data_df')
        save_dataframe_to_hdf5(hdf, legend_df, 'legend_df')

        if config is not None:
            # Save config as attributes
            for key, value in config.items():
                hdf.attrs[key] = value


def load_dr_cell_h5(file: str) -> (pd.DataFrame, pd.DataFrame, dict):
    with h5py.File(file, 'r') as hdf:
        data_df = load_dataframe_from_hdf5(hdf, 'data_df')
        legend_df = load_dataframe_from_hdf5(hdf, 'legend_df')

        # Read config from attributes
        config = {key: hdf.attrs[key] for key in hdf.attrs}
        if config is None:
            with open('default_file_config.json', 'r') as json_file:
                config = json.load(json_file)

    return data_df, legend_df, config


def validate_drcell_file(file: str) -> bool:
    try:
        data_df, legend_df, config = load_dr_cell_h5(file)
    except Exception:
        return False
    if not (isinstance(data_df, pd.DataFrame) and isinstance(legend_df, pd.DataFrame) and (
            config is None or isinstance(config, dict))):
        return False

    # if config is not None and (
    #         "recording_type" not in config.keys() or "data_variables" not in config.keys() or "display_hover_variables" not in config.keys()):
    #     return False
    # # checks if all the variables in the config are in the dataframes
    # if config is not None and (all(value in config["data_variables"] for value in legend_df.columns.tolist()) or all(
    #         value in config["display_hover_variables"] for value in legend_df.columns.tolist())):
    #     print("Invalid config")
    #     return False

    return True


def determine_column_type(column_data):
    """
    Determine the type of the column data.
    """
    if column_data.dropna().apply(lambda x: isinstance(x, str)).all():
        return 'string'
    if column_data.dropna().apply(lambda x: isinstance(x, bool)).all():
        return 'boolean'
    try:
        pd.to_numeric(column_data, errors='raise')
        return 'numeric'
    except ValueError:
        return 'mixed'  # If neither string nor numeric


def save_dataframe_to_hdf5(h5file, df, name):
    group = h5file.create_group(name)
    for i, col in enumerate(df.columns):
        column_data = df[col]
        column_type = determine_column_type(column_data)

        # Use column index as name if no column name is provided
        col_name = str(col) if col is not None else f"col_{i}"

        if column_type == 'string':
            column_data = column_data.fillna('')
            dtype = h5py.special_dtype(vlen=str)
            dataset = group.create_dataset(col_name, data=column_data.values, dtype=dtype)
        elif column_type == 'numeric':
            column_data = pd.to_numeric(column_data, errors='coerce')
            dataset = group.create_dataset(col_name, data=column_data.values)
        elif column_type == 'boolean':
            column_data = column_data.map({True: 'True', False: 'False'}).fillna('NaN')
            dtype = h5py.special_dtype(vlen=str)
            dataset = group.create_dataset(col_name, data=column_data.values, dtype=dtype)
        else:
            column_data = column_data.fillna('')
            dtype = h5py.special_dtype(vlen=str)
            dataset = group.create_dataset(col_name, data=column_data.values, dtype=dtype)

        # Save the type information as metadata
        dataset.attrs['dtype'] = column_type


def load_dataframe_from_hdf5(h5file, name):
    group = h5file[name]
    data = {}
    for col in group:
        dataset = group[col]
        dtype = dataset.attrs['dtype']  # Retrieve the type information from metadata

        if dtype == 'string':
            values = [s.decode('utf-8') if isinstance(s, bytes) else s for s in dataset[:]]
            data[col] = pd.Series(values).replace('', np.nan).values
        else:
            values = dataset[:]
            if dtype == 'numeric':
                if np.issubdtype(values.dtype, np.floating):
                    data[col] = np.where(np.isnan(values), np.nan, values)
                elif np.issubdtype(values.dtype, np.integer):
                    data[col] = values.astype(int)
            elif dtype == 'boolean':
                values = [v.decode('utf-8') if isinstance(v, bytes) else v for v in values]
                data[col] = pd.Series(values).replace({'NaN': np.nan, 'True': True, 'False': False}).values

    df = pd.DataFrame(data)
    return df


def load_and_preprocess_mat_data_AD_IL(c_file: str, recording_type: str = '2P'):
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


def convert_data_AD_IL(input_file_path: str, output_path: str, recording_type: str = None):
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


# TODO work in progress, not final yet
def load_and_preprocess_example_mat_data(c_file: str):
    # load data
    example_mat_file = sio.loadmat(c_file)
    data_array = np.concatenate((example_mat_file["neuron_PSTH_lick_left_correct"],
                                 example_mat_file["neuron_PSTH_lick_right_correct"]), axis=1)
    matrix_array = np.concatenate((example_mat_file["neuron_info_cell_type"],
                                   example_mat_file["neuron_info_photoinhibition"],
                                   example_mat_file["neuron_info_activity_mode_w"],
                                   example_mat_file["neuron_info_connectivity"],
                                   example_mat_file["neuron_info_depth"],
                                   example_mat_file["neuron_info_mice_session"]), axis=1)
    data_df = pd.DataFrame(data_array)
    matrix_df = pd.DataFrame(matrix_array,
                             columns=["cell_type", "photoinhibition", "activity_mode_w_c0", "activity_mode_w_c1",
                                      "activity_mode_w_c2", "activity_mode_w_c3", "activity_mode_w_c4",
                                      "activity_mode_w_c5", "connectivity_c0", "connectivity_c1", "connectivity_c2",
                                      "depth", "mice_session_c0", "mice_session_c1", "mice_session_c2",
                                      "mice_session_c3"])
    # variables from the input data, that is selectable in the Color and Filter setting
    data_variables = ["cell_type", "photoinhibition", "activity_mode_w_c0", "activity_mode_w_c1",
                      "activity_mode_w_c2", ]
    # variables from the input data, that gets displayed in the hover tool
    display_hover_variables = ["pdIndex", "depth", "cell_type", "photoinhibition"]
    config = {
        "recording_type": "2P",
        "data_variables": data_variables,
        "display_hover_variables": display_hover_variables,
    }
    return data_df, matrix_df, config
    print(g)

    # g = h5py.File(cFile,'r')
    # X = np.array(g['X'])
