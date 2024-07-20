import json
import os
import sys

from bokeh.palettes import Paired12

project_path = r"C:\path\to\DrCELL"
# change working directory, because Bokeh Server doesn't recognize it otherwise
os.chdir(os.path.join(project_path))
# add to project Path so Bokeh Server can import other python files correctly
sys.path.append(project_path)

import drcell.dimensionalReduction.phate
import drcell.dimensionalReduction.tsne
import drcell.dimensionalReduction.umap
import drcell.drCELLBrokehApplication
import drcell.util.drCELLFileUtil

debug = False
experimental = True
bokeh_show = False
color_palette = Paired12

data_path = os.path.join(project_path, "data")
output_path = data_path
# it's important to use different names for different datasets!
# otherwise already buffered data from older datasets gets mixed with the new dataset!


included_datasets = [
    # "2P_Test_all.h5",
    "AllDataMMMatrixZscoredv3_all.h5",
]

included_legacy_matlab_datasets = [
    # ("20240313_091532_MedianChoiceStim30trials_AllTasks_ForLuca.mat", "2P"),
    ("2P_Test.mat", "2P"),
    # ("AllDataMMMatrixZscoredv3.mat", "Ephys"),
    # ("AllDataMMMatrixZscoredBin1.mat", "Ephys")
]

input_file_paths = []
for dataset in included_datasets:
    input_file_paths.append(os.path.join(data_path, dataset))

for matlab_dataset in included_legacy_matlab_datasets:
    input_matlab_file_path = os.path.join(data_path, matlab_dataset[0])
    recording_type = matlab_dataset[1]

    print(f"Converting {input_matlab_file_path} to DrCELL .h5 files")
    converted_input_file_paths = drcell.util.drCELLFileUtil.convert_data_AD_IL(input_matlab_file_path,
                                                                               os.path.dirname(input_matlab_file_path),
                                                                               recording_type=recording_type)
    input_file_paths.extend(converted_input_file_paths)

# checks if there is a image server port given in the arguments; if not defaults to 8000
image_server_port = int(sys.argv[1]) if len(sys.argv) > 1 else '8000'

# loads parameters and default values from config file; out of box functions get assigned additionally
with open('config/reduction_functions_config.json', 'r') as json_file:
    reduction_functions = json.load(json_file)

reduction_functions["UMAP"]["function"] = drcell.dimensionalReduction.umap.generate_umap
reduction_functions["UMAP"]["diagnostic_functions"] = drcell.dimensionalReduction.umap.generate_umap_diagnostic_plot

reduction_functions["t-SNE"]["function"] = drcell.dimensionalReduction.tsne.generate_t_sne
reduction_functions["t-SNE"]["diagnostic_functions"] = None

reduction_functions["PHATE"]["function"] = drcell.dimensionalReduction.phate.generate_phate
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


if __name__ == '__main__':
    bokeh_show = True

# loads bokeh interface
drcell.drCELLBrokehApplication.plot_bokeh(input_file_paths, reduction_functions=reduction_functions,
                                          bokeh_show=bokeh_show, color_palette=color_palette, debug=debug,
                                          experimental=experimental, output_path=output_path,
                                          image_server_port=image_server_port)
