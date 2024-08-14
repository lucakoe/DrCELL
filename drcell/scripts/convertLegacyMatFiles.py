import os

import drcell

if __name__ == "__main__":
    # Parse command-line arguments
    # parser = argparse.ArgumentParser(description="Convert Legacy .mat files to DrCell .h5 files")
    # parser.add_argument("--type", type=str, default=None, help="Type of recording (Ephys, 2P or None)")

    # TODO finish integrating into CLI
    data_path = r"C:\path\to\datafolder"
    included_legacy_matlab_datasets = [
        # ("20240313_091532_MedianChoiceStim30trials_AllTasks_ForLuca.mat", "2P"),
        ("2P_Test.mat", "2P"),
        # ("AllDataMMMatrixZscoredv3.mat", "Ephys"),
        # ("AllDataMMMatrixZscoredBin1.mat", "Ephys")
    ]

    for matlab_dataset in included_legacy_matlab_datasets:
        input_matlab_file_path = os.path.join(data_path, matlab_dataset[0])
        recording_type = matlab_dataset[1]

        print(f"Converting {input_matlab_file_path} to DrCELL .h5 files")
        converted_input_file_paths = drcell.util.drCELLFileUtil.convert_data_AD_IL(input_matlab_file_path,
                                                                                   os.path.dirname(
                                                                                       input_matlab_file_path),
                                                                                   recording_type=recording_type)
        print(f"Converted files: {converted_input_file_paths}")

    # args = parser.parse_args()
