import json
import os

import h5py
import numpy as np
import pandas as pd
from scipy import io as sio


def save_as_dr_cell_h5(file, data_df, legend_df, config=None):
    # TODO maybe change to easier hdf5 structure for direct matlab export
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
