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
image_server_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

if __name__ == '__main__':
    bokeh_show = True

# loads bokeh interface
drcell.drCELLBrokehApplication.plot_bokeh(input_file_paths, reduction_functions=None,
                                          bokeh_show=bokeh_show, color_palette=color_palette, debug=debug,
                                          experimental=experimental, output_path=output_path,
                                          image_server_port=image_server_port)
