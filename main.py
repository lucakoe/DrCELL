import os
import sys
import hdbscan
import numpy as np
import pandas
# import h5py
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from bokeh.palettes import Spectral4, Spectral11, Turbo256, Spectral10, Paired12, Muted9
import util

# skips Spike Plot generation if false and takes dump
generate_and_save_spike_plots = False
generate_diagnostic_plots = False
generate_umap_parameters = False
debug = False
experimental = True
bokeh_show = False
color_palette = Paired12
# variables from the input data, that is selectable in the Color and Filter setting
data_variables = ["Task", "IsChoiceSelect", "IsStimSelect", "RedNeurons"]
# variables from the input data, that gets displayed in the hover tool
display_hover_variables = ["pdIndex", "Neuron", "ChoiceAUCs", "StimAUCs"]

project_path = r"C:\Users\Luca Koenig\OneDrive - Students RWTH Aachen University\Bachelorarbeit\GitHub\CELL"
data_path = os.path.join(project_path, "data")
# it's important to use different names for different datasets!
# otherwise already buffered data from older datasets gets mixed with the new dataset!
input_file_path = os.path.join(data_path, r'Umap_2530_2532MedianChoiceStim30trials_Array.mat')
# path created for the data of the given input file
input_folder_path = os.path.join(data_path, os.path.splitext(os.path.basename(input_file_path))[0])
# folder for output file for exports of corresponding input file
output_path = os.path.join(input_folder_path, "output")
# folder for spike plot images of corresponding input file
spike_plot_images_path = os.path.join(input_folder_path, "plot_images")
# folder for buffered UMAP data of corresponding input file
dump_files_path = os.path.join(input_folder_path, "dump")
# takes the input file name as name for dumpfile
umap_out_param_dump_filename_extension = "_parameter_buffer_dump.pkl"

reduction_functions = {}
# n_components has to be 2 in custom functions; first parameter has to be data!!!
reduction_functions["UMAP"] = {"function": util.generate_umap,
                               "numeric_parameters": {"n_neighbors": {"start": 2, "end": 50, "step": 1, "value": 20},
                                                      "min_dist": {"start": 0.00, "end": 1.0, "step": 0.01,
                                                                   "value": 0.0}},
                               "bool_parameters": {},
                               "select_parameters": {},
                               "constant_parameters": {"n_components": (2), "random_state": (42)}}
reduction_functions["t-SNE"] = {"function": util.generate_t_sne,
                                "numeric_parameters": {"perplexity": {"start": 5, "end": 50, "step": 1, "value": 30},
                                                       "learning_rate": {"start": 10, "end": 200, "step": 10,
                                                                         "value": 200},
                                                       "n_iter": {"start": 250, "end": 1000, "step": 10, "value": 1000},
                                                       "early_exaggeration": {"start": 4, "end": 20, "step": 1,
                                                                              "value": 12},
                                                       "angle": {"start": 0.2, "end": 0.8, "step": 0.1, "value": 0.5}},
                                "bool_parameters": {},
                                "select_parameters": {
                                    "metric": {"options": ["euclidean", "manhattan", "cosine"],
                                               "default_option": "euclidean"}},
                                "constant_parameters": {"n_components": (2)}}

reduction_functions["PHATE"] = {"function": util.generate_phate,
                                "numeric_parameters": {"knn": {"start": 5, "end": 100, "step": 1, "value": 30},
                                                       "decay": {"start": 1, "end": 50, "step": 1,
                                                                 "value": 15},
                                                       "t": {"start": 5, "end": 100, "step": 1, "value": 5},
                                                       "gamma": {"start": 0, "end": 10, "step": 0.1,
                                                                 "value": 0},
                                                       "n_jobs": {"start": -1, "end": 4, "step": 1, "value": -1},
                                                       "n_pca": {"start": 5, "end": 100, "step": 1,
                                                                 "value": 100},
                                                       "n_landmark": {"start": 50, "end": 1000, "step": 10,
                                                                      "value": 1000},

                                                       },
                                "bool_parameters": {"verbose": False},
                                "select_parameters": {},
                                "constant_parameters": {"n_components": (2)}}

print(f"Using '{input_file_path}' as input file.")
# creates a folder for the corresponding input file, where data gets saved
if not os.path.exists(input_folder_path):
    # If not, create it
    os.makedirs(input_folder_path)
    os.makedirs(spike_plot_images_path)
    os.makedirs(dump_files_path)
    os.makedirs(output_path)
    print(f"Folder '{input_folder_path}' created.")
else:
    if not os.path.exists(spike_plot_images_path):
        os.makedirs(spike_plot_images_path)
    if not os.path.exists(dump_files_path):
        os.makedirs(dump_files_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"Folder '{input_folder_path}' already exists.")

# change working directory, because Bokeh Server doesn't recognize it otherwise
os.chdir(os.path.join(project_path))
# add to project Path so Bokeh Server can import other python files correctly
sys.path.append(project_path)

import plotting
import util

# ['C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate2530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixTaskWithoutDelay2530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixTaskWithDelay2530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate22530.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate2532.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixAudioTaskWithoutDelay2532.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixAudioTaskWithDelay2532.mat', 'C:\\Users\\despatin\\Downloads\\UMAPmatrixInnate22532.mat']
# 
# allFiles = ['Q:\\BpodImager\\umapClust\\20210309T2356_fullA.mat'] #first file

if __name__ == '__main__':
    bokeh_show = True

titles = ["all", "excludeChoiceUnselectBefore", "excludeStimUnselectBefore"]
umap_df, matrix_legend_df, cleaned_data = util.load_and_preprocess_data(input_file_path)

cleaned_datas = {}
matrix_legend_dfs = {}
dump_files_paths = {}
cell_dfs = {}

for title in titles:
    cleaned_datas[title] = cleaned_data
    matrix_legend_dfs[title] = matrix_legend_df

    if title == "all":
        print(f"{title} Data Length: {len(matrix_legend_df)}")

    elif title == "excludeChoiceUnselectBefore":
        # Filters cells with Property
        cleaned_datas[title] = cleaned_data[matrix_legend_df["IsChoiceSelect"]]
        matrix_legend_dfs[title] = matrix_legend_df[matrix_legend_df["IsChoiceSelect"]]
        print(f"{title} Data Length: {matrix_legend_df['IsChoiceSelect'].apply(lambda x: x).sum()}")

    elif title == "excludeStimUnselectBefore":
        # Filters cells with Property
        cleaned_datas[title] = cleaned_data[matrix_legend_df["IsStimSelect"]]
        matrix_legend_dfs[title] = matrix_legend_df[matrix_legend_df["IsStimSelect"]]
        print(f"{title} Data Length: {matrix_legend_df['IsStimSelect'].apply(lambda x: x).sum()}")

    if debug: print(f"Cleaned Data {title}: \n{cleaned_datas[title]}")

    dump_files_paths[title] = os.path.abspath(
        os.path.join(dump_files_path, title + umap_out_param_dump_filename_extension))

    # Convert the Matplotlib plots to base64-encoded PNGs

    if generate_and_save_spike_plots and title == "all":
        for i in range(umap_df.shape[0]):
            plotting.plot_and_save_spikes(i, umap_df, spike_plot_images_path, fps=30,
                                          number_consecutive_recordings=2)

    temp_umap_out = pandas.DataFrame
    umap_object = umap.UMAP

    if generate_umap_parameters:
        if experimental:
            n_neighbors_values = range(2, 51, 1)
            min_dist_values = np.arange(0.0, 1.01, 0.01).tolist()
            min_dist_values = [round(x, 2) for x in min_dist_values]
            pca_n_components = range(2, cleaned_datas[title].shape[1])

        else:
            n_neighbors_values = range(10, 21, 1)
            min_dist_values = np.arange(0.0, 0.11, 0.01).tolist()
            min_dist_values = [round(x, 2) for x in min_dist_values]
            # TODO add pca_n_component range
            pca_n_components = []

        util.generate_umap_parameters(cleaned_datas[title], dump_files_paths[title], n_neighbors_values,
                                      min_dist_values, pca_n_components)
    # gets the UMAP Output file. This function is used to buffer already created UMAPs and improve performance
    if debug: print(os.path.abspath(dump_files_paths[title]))
    umap_default_parameters = {"n_neighbors": 20, "min_dist": 0.0, "n_components": 2}
    temp_umap_out = util.get_dimensional_reduction_out("UMAP", cleaned_datas[title],
                                                       dump_path=os.path.abspath(dump_files_paths[title]).replace(
                                                           "\\", '/'),
                                                       reduction_functions=reduction_functions,
                                                       reduction_params=umap_default_parameters,
                                                       pca_preprocessing=False)

    if debug: print('Umap vals: ' + str(temp_umap_out.shape))

    # Apply HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1)
    clusters = clusterer.fit_predict(temp_umap_out)

    cell_dfs[title] = pd.DataFrame(temp_umap_out, columns=['x', 'y'])
    # creates an index for merging
    cell_dfs[title].index = range(len(cell_dfs[title]))
    matrix_legend_dfs[title].index = range(len(matrix_legend_dfs[title]))
    cell_dfs[title] = cell_dfs[title].merge(matrix_legend_dfs[title], left_index=True, right_index=True)
    cell_dfs[title]['Task'] = cell_dfs[title]['Task'].astype(str)
    # Add cluster labels to your dataframe
    cell_dfs[title]['Cluster'] = clusters

# plots UMAP via Bokeh
plotting.plot_bokeh(cell_dfs, cleaned_datas, spike_plot_images_path, dump_files_paths, titles,
                    reduction_functions=reduction_functions,
                    bokeh_show=bokeh_show,
                    start_dropdown_data_option=titles[0], data_variables=data_variables,
                    display_hover_variables=display_hover_variables, color_palette=color_palette, debug=debug,
                    experimental=experimental, output_file_path=output_path)

# # show results
# cIdx = np.random.permutation(matrixLegendDf["Task"].size)
# fig1, ax1 = plt.subplots()
# # cColor = sns.color_palette("Dark2", np.max([groups])+1);
# cColor = sns.color_palette("Spectral", np.max([matrixLegendDf["Task"].astype(float).astype(int)]) + 1);
# ax1.scatter(tempUmapOut[cIdx, 0], tempUmapOut[cIdx, 1], c=[cColor[x] for x in matrixLegendDf["Task"][cIdx]], s=2)
# # plt.axis('square')
# ax1.set_aspect('equal', adjustable='datalim')
# plt.show()

# sio.savemat(cFile.replace('.mat','_umap_allAreas.mat'),
#                 {'umapOut':umapOut,
#                  'tempUmapOut':tempUmapOut,
#                  })

# Save UMAP coordinates and labels to see in Matlab
# umap_data = np.column_stack((tempUmapOut, matrixLegendDf["Task"]))
# file_name = os.path.basename(cFile)
# sio.savemat(r'\\Naskampa\lts\2P_PuffyPenguin\2P_PuffyPenguin\Umap_Analysis\output_' + file_name, {'umap_data': umap_data})

if generate_diagnostic_plots:
    (plotting.generate_diagnostic_plots(umap_object, cleaned_data))
