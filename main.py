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

project_path = r"C:\Users\koenig\OneDrive - Students RWTH Aachen University\Bachelorarbeit\GitHub\CELL"
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
umap_out_param_dump_filename_extension = "_umapOutParamDump.pkl"

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
            min_dist_values = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13,
                               0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27,
                               0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41,
                               0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55,
                               0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
                               0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83,
                               0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97,
                               0.98, 0.99, 1.00]
            pca_n_components = range(2, cleaned_datas[title].shape[1])

        else:
            n_neighbors_values = range(10, 21, 1)
            min_dist_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
            # TODO add pca_n_component range
            pca_n_components = []

        n_components_values = [2]
        for n_neighbors_value in n_neighbors_values:
            for min_dist_value in min_dist_values:
                for n_components_value in n_components_values:
                    util.get_umap_out(cleaned_datas[title],
                                      os.path.abspath(dump_files_paths[title]),
                                      n_neighbors=n_neighbors_value,
                                      min_dist=min_dist_value, n_components=n_components_value)
                    if n_components_value > 3:
                        for pca_n_components_value in pca_n_components:
                            util.get_umap_out(cleaned_datas[title],
                                              os.path.abspath(os.path.abspath(dump_files_paths[title])).replace("\\",
                                                                                                                '/'),
                                              n_neighbors=n_neighbors_value, min_dist=round(min_dist_value, 2),
                                              n_components=n_components_value,
                                              pca_n_components=int(pca_n_components_value),
                                              pca_preprocessing=True)
    # gets the UMAP Output file. This function is used to buffer already created UMAPs and improve performance
    if debug: print(os.path.abspath(dump_files_paths[title]))
    temp_umap_out = util.get_umap_out(cleaned_datas[title],
                                      os.path.abspath(dump_files_paths[title]).replace(
                                          "\\", '/'),
                                      n_neighbors=20, min_dist=0.0, n_components=2)

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
