import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bokeh.palettes import Spectral4, Spectral11, Turbo256, Spectral10, Paired12, Muted9
import util

debug = False
experimental = True
bokeh_show = False
color_palette = Paired12

project_path = r"C:\Users\koenig\OneDrive - Students RWTH Aachen University\Bachelorarbeit\GitHub\CELL"
data_path = os.path.join(project_path, "data")
# it's important to use different names for different datasets!
# otherwise already buffered data from older datasets gets mixed with the new dataset!
input_file_path = os.path.join(data_path, r'AllDataMMMatrixZscoredBin1.mat')
dataset_type = "Ephys"

# path created for the data of the given input file
input_folder_path = os.path.join(data_path, os.path.splitext(os.path.basename(input_file_path))[0])
# folder for output file for exports of corresponding input file
output_path = os.path.join(input_folder_path, "output")
# folder for buffered UMAP data of corresponding input file
dump_files_path = os.path.join(input_folder_path, "dump")
# takes the input file name as name for dumpfile
output_buffer_param_dump_filename_extension = "_parameter_buffer_dump.pkl"

data_variables = []
display_hover_variables = []
if dataset_type == "Ephys":
    # variables from the input data, that is selectable in the Color and Filter setting
    data_variables = ["IsChoiceSel", "IsStimSel", "Area", "IsStimSelVisual", "IsStimSelTactile",
                      "IsStimSelMultisensory",
                      "IsChoiceSelVisual", "IsChoiceSelTactile", "IsChoiceSelMultisensory"]
    # variables from the input data, that gets displayed in the hover tool
    display_hover_variables = ["pdIndex", "Neuron", "Area", "ChoiceAUCsVisual", "ChoiceAUCsTactile",
                               "ChoiceAUCsMultisensory", "StimAUCsVisual", "StimAUCsTactile", "StimAUCsMultisensory"]
elif dataset_type == "2P":
    # variables from the input data, that is selectable in the Color and Filter setting
    data_variables = ["IsChoiceSelect", "IsStimSelect", "Task", "RedNeurons"]
    # variables from the input data, that gets displayed in the hover tool
    display_hover_variables = ["pdIndex", "Neuron", "ChoiceAUCs", "StimAUCs"]

# checks if there is a image server port given in the arguments; if not defaults to 8000
image_server_port = int(sys.argv[1]) if len(sys.argv) > 1 else '8000'

# loads parameters and default values from config file; out of box functions get assigned additionally
with open('reduction_functions_config.json', 'r') as json_file:
    reduction_functions = json.load(json_file)

reduction_functions["UMAP"]["function"] = util.generate_umap
reduction_functions["UMAP"]["diagnostic_functions"] = util.generate_umap_diagnostic_plot

reduction_functions["t-SNE"]["function"] = util.generate_t_sne
reduction_functions["t-SNE"]["diagnostic_functions"] = None

reduction_functions["PHATE"]["function"] = util.generate_phate
reduction_functions["PHATE"]["diagnostic_functions"] = None
# Example custom function
# # n_components has to be 2 in custom functions; first parameter has to be data!!!
# reduction_functions["custom_function_example"] = {"function": util.custom_function,
#                                 "diagnostic_functions": {"diagnostic_plots": util.custom_diagnostic_function},
#                                 "numeric_parameters": {"numeric_example_parameter_1": {"start": 5, "end": 50, "step": 1, "value": 30},
#                                                        "numeric_example_parameter_2": {"start": 10, "end": 200, "step": 10,
#                                                                          "value": 200},
#                                                        "numeric_example_parameter_n": {"start": 250, "end": 1000, "step": 10, "value": 1000},
#                                                        },
#                                 "bool_parameters": {"bool_example_parameter_1": False, "bool_example_parameter_2": True, "bool_example_parameter_n": False},
#                                 "nominal_parameters": {
#                                     "nominal_example_parameter_1": {"options": ["example_option_1", "example_option_2", "example_option_n"],
#                                                                    "default_option": "example_option_2"},
#                                     "nominal_example_parameter_2": {
#                                         "options": ["example_option_1", "example_option_2", "example_option_n"],
#                                         "default_option": "example_option_1"},
#                                     "nominal_example_parameter_n": {
#                                         "options": ["example_option_1", "example_option_2", "example_option_n"],
#                                         "default_option": "example_option_n"}
#                                 },
#                                                   "constant_parameters": {"n_components": (2),
#                                                                           "constant_parameter_1": ("example_parameter"),
#                                                                           "constant_parameter_2": (5),
#                                                                           "constant_parameter_n": (True)}}

print(f"Using '{input_file_path}' as input file.")
# creates a folder for the corresponding input file, where data gets saved
if not os.path.exists(input_folder_path):
    # If not, create it
    os.makedirs(input_folder_path)
    os.makedirs(dump_files_path)
    os.makedirs(output_path)
    print(f"Folder '{input_folder_path}' created.")
else:
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
umap_df, matrix_legend_df, cleaned_data = util.load_and_preprocess_data(input_file_path, dataset_type=dataset_type)

cleaned_data_arrays = {}
matrix_legend_dfs = {}
dump_files_paths = {}
recording_types = {}

# filter out specific values beforehand and add them as seperate datasets
for title in titles:
    cleaned_data_arrays[title] = cleaned_data
    matrix_legend_dfs[title] = matrix_legend_df
    recording_types[title] = dataset_type
    if title == "all":
        print(f"{title} Data Length: {len(matrix_legend_df)}")

    elif title == "excludeChoiceUnselectBefore":
        # Filters cells with Property
        cleaned_data_arrays[title] = cleaned_data[matrix_legend_df[data_variables[0]]]
        matrix_legend_dfs[title] = matrix_legend_df[matrix_legend_df[data_variables[0]]]
        print(f"{title} Data Length: {matrix_legend_df[data_variables[0]].apply(lambda x: x).sum()}")

    elif title == "excludeStimUnselectBefore":

        # Filters cells with Property
        cleaned_data_arrays[title] = cleaned_data[matrix_legend_df[data_variables[1]]]
        matrix_legend_dfs[title] = matrix_legend_df[matrix_legend_df[data_variables[1]]]
        print(f"{title} Data Length: {matrix_legend_df[data_variables[1]].apply(lambda x: x).sum()}")

    if debug: print(f"Cleaned Data {title}: \n{cleaned_data_arrays[title]}")

    dump_files_paths[title] = os.path.abspath(
        os.path.join(dump_files_path, title + output_buffer_param_dump_filename_extension))

# loads bokeh interface
plotting.plot_bokeh(cleaned_data_arrays, matrix_legend_dfs, dump_files_paths, titles, recording_types=recording_types,
                    reduction_functions=reduction_functions,
                    bokeh_show=bokeh_show,
                    start_dropdown_data_option=titles[0], data_variables=data_variables,
                    display_hover_variables=display_hover_variables, color_palette=color_palette, debug=debug,
                    experimental=experimental, output_file_path=output_path, image_server_port=image_server_port)
