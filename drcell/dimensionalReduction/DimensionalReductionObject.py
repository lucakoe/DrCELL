import os
import pickle
from abc import abstractmethod

import numpy as np
import sklearn.decomposition
from matplotlib import pyplot as plt



class DimensionalReductionObject:

    def __init__(self, name: str, parms:dict, diagnostic_functions:dict ):
        self.name = name
        self.diagnostic_functions = diagnostic_functions

    @classmethod
    def from_json_config(cls, name: str, function, json_config, diagnostic_function=None):
        # config_dict
        # config_dict['function'] = function
        # reduction_functions["PHATE"]["diagnostic_functions"] =
        #
        # return cls(value1, config_dict)
        pass

    @abstractmethod
    def reduce_dimensions(self, data):
        pass

    def diagnostic_function(self, name:str):
        return self.diagnostic_functions[name]
    def list_diagnostic_functions_names(self) -> list:
        return list(self.diagnostic_functions.keys())

    def generate_config_json(self, output_file_path):
        pass

    def get_dimensional_reduction_function_dict(self):
        pass

    def get_dimensional_reduction_function_dict_entry(self):
        pass

    def __str__(self):
        return str(self.get_dimensional_reduction_function_dict())


dimensional_reduction_out_dump_data = {}


def apply_pca_preprocessing(data, n_components=2, show_diagnostic_plot=False):
    """
    Apply PCA preprocessing to the input data.

    Parameters:
    - data: The input data.
    - n_components: The number of components for PCA.
    - show_diagnostic_plot: Whether to show the PCA diagnostic plot.

    Returns:
    - The PCA preprocessed data.
    """
    pca_operator = sklearn.decomposition.PCA(n_components=n_components)
    pca_data = pca_operator.fit_transform(data)

    if show_diagnostic_plot:
        diagnostic_data = pca_operator.explained_variance_ratio_
        diagnostic_plot = return_pca_diagnostic_plot(diagnostic_data)
        diagnostic_plot.show()

    return pca_data


def get_dimensional_reduction_out(reduction_function_name, data, dump_folder_path, reduction_functions,
                                  reduction_params,
                                  pca_preprocessing=False, pca_n_components=2, show_pca_diagnostic_plot=False,
                                  output_buffer_param_dump_filename_extension="_parameter_buffer_dump.pkl"):
    dataset_name = os.path.basename(dump_folder_path)
    dump_folder_path = os.path.join(dump_folder_path,
                                    f"{reduction_function_name}" + output_buffer_param_dump_filename_extension)

    if reduction_function_name == "None" or reduction_function_name is None:
        pca_preprocessing = True
        pca_n_components = 2
        param_key = (("pca_preprocessing_n_components", pca_n_components),)
        reduction_params = {"pca_preprocessing_n_components": pca_n_components}
    else:
        # sort parameters alphabetically, to prevent duplicates in dump file
        reduction_params_items = list(reduction_params.items())
        reduction_params_items.sort()
        if pca_preprocessing:
            param_key = tuple(reduction_params_items) + (("pca_preprocessing_n_components", pca_n_components),)
        else:
            param_key = tuple(reduction_params_items)
    buffered_data_dump = {}

    # Checks if the dump file with this path was already called.
    # If so, instead of loading it for every call of the function, it takes the data from there
    if dump_folder_path not in dimensional_reduction_out_dump_data:
        # Check if the file exists
        if os.path.exists(dump_folder_path):
            with open(dump_folder_path, 'rb') as file:
                dimensional_reduction_out_dump_data[dump_folder_path] = pickle.load(file)
        else:
            # If the file doesn't exist, create it and write something to it
            with open(dump_folder_path, 'wb') as file:
                pickle.dump(buffered_data_dump, file)
                dimensional_reduction_out_dump_data[dump_folder_path] = buffered_data_dump

            print(f"The file '{dump_folder_path}' has been created.")

    buffered_data_dump = dimensional_reduction_out_dump_data[dump_folder_path]
    current_data = data

    if param_key not in buffered_data_dump:
        if pca_preprocessing:
            print(
                f"Generate {reduction_function_name} with PCA preprocessing: File = {dataset_name}/{os.path.basename(dump_folder_path)}, {reduction_params}, PCA n_components = {pca_n_components}")
            current_data = apply_pca_preprocessing(current_data, n_components=pca_n_components,
                                                   show_diagnostic_plot=show_pca_diagnostic_plot)
        else:
            print(
                f"Generate {reduction_function_name}: File = {dataset_name}/{os.path.basename(dump_folder_path)}, {reduction_params}")

        if reduction_function_name == "None" or reduction_function_name == None:
            reduced_data = current_data
        else:
            reduced_data = reduction_functions[reduction_function_name]["function"](current_data, **reduction_params)

        buffered_data_dump[param_key] = reduced_data

        with open(dump_folder_path, 'wb') as file:
            dimensional_reduction_out_dump_data[dump_folder_path] = buffered_data_dump
            pickle.dump(buffered_data_dump, file)

    print(
        f"Return {reduction_function_name}: File = {dataset_name}/{os.path.basename(dump_folder_path)}, {reduction_params}")
    return buffered_data_dump[param_key]


def return_pca_diagnostic_plot(diagnostic_data):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the bar graph
    ax.bar(np.arange(len(diagnostic_data)), diagnostic_data, label='Individual Values', color='blue',
           alpha=0.7)

    # Plot the cumulative line
    cumulative_values = np.cumsum(diagnostic_data)
    ax.plot(np.arange(len(diagnostic_data)), cumulative_values, label='Cumulative Values', color='red',
            linestyle='--',
            marker='o')

    # Add labels and title
    ax.set_xlabel('Components')
    ax.set_ylabel('Explained Variance')
    ax.set_title('PCA Diagnostic Plot')
    ax.legend()

    return plt
