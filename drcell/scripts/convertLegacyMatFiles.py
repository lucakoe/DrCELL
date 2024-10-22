import argparse
import glob
import os
import sys

import drcell


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert Legacy .mat files to DrCell .h5 files")
    parser.add_argument("file_or_folder_path", type=str, default=sys.argv[0],
                        help="Path to the DrCELL file or folder containing the DrCELL files.")
    parser.add_argument("--type", type=str, default=None, help="Type of recording(s) (Ephys, 2P or None)")

    args = parser.parse_args()
    mat_file_paths = []
    if os.path.isdir(args.file_or_folder_path):
        for path in glob.glob(os.path.join(args.file_or_folder_path, '*.mat')):
            mat_file_paths.append(os.path.abspath(path))
    elif os.path.isfile(args.file_or_folder_path):
        mat_file_paths = [args.file_or_folder_path]

    for matlab_dataset in mat_file_paths:
        recording_type = args.type

        print(f"Converting {os.path.basename(matlab_dataset)} to DrCELL .h5 files")
        converted_input_file_paths = drcell.util.drCELLFileUtil.convert_data_AD_IL(matlab_dataset,
                                                                                   os.path.dirname(
                                                                                       matlab_dataset),
                                                                                   recording_type=recording_type)
        print(f"Converted files: {converted_input_file_paths}")


if __name__ == '__main__':
    main()
