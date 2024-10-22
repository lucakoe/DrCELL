import argparse
import os
import sys

import drcell


def main():
    parser = argparse.ArgumentParser(description="Convert Legacy .mat files to DrCell .h5 files")
    parser.add_argument("example_mat_file_path", type=str, default=sys.argv[0],
                        help="Path to the DrCELL file or folder containing the DrCELL files.")

    args = parser.parse_args()

    data_df, matrix_df, config = drcell.load_and_preprocess_example_mat_data(args.example_mat_file_path)
    drcell.save_as_dr_cell_h5(os.path.join(os.path.dirname(args.example_mat_file_path), "example_drcell.h5"), data_df,
                              matrix_df, config)


if __name__ == '__main__':
    main()
