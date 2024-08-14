import os
import pickle
from abc import abstractmethod

import numpy as np
import sklearn.decomposition
from matplotlib import pyplot as plt

from drcell.util.generalUtil import get_decimal_places


class DimensionalReductionObject:
    @staticmethod
    def apply_pca_preprocessing(data, n_components: int = 2, show_diagnostic_plot: bool = False):
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
            diagnostic_plot = DimensionalReductionObject.return_pca_diagnostic_plot(diagnostic_data)
            diagnostic_plot.show()

        return pca_data

    @staticmethod
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

    def __init__(self, name: str, params: dict, diagnostic_functions: dict = None):
        """
        Initializes a DimensionalReductionObject.

        Args:
            name (str): The name of the object.
            params (dict): The parameters for the object.
            diagnostic_functions (dict, optional): The diagnostic functions for the object. Defaults to None.
        """
        self.name = name
        self.params = params
        if diagnostic_functions is None:
            self.diagnostic_functions = {}
        else:
            self.diagnostic_functions = diagnostic_functions
        # default params file
        self.build_in_params = self.params
        self.dimensional_reduction_out_dump_data = {}

    @classmethod
    def from_json_config(cls, name: str, function, json_config, diagnostic_function=None):
        # config_dict
        # config_dict['function'] = function
        # reduction_functions["PHATE"]["diagnostic_functions"] =
        #
        # return cls(value1, config_dict)
        pass

    def get_default_params(self) -> dict:
        default_params = {}
        for parameter in self.params["numerical_parameters"].keys():
            default_params[parameter] = self.params["numerical_parameters"][parameter]["value"]
        for parameter in self.params["bool_parameters"].keys():
            default_params[parameter] = self.params["bool_parameters"][parameter]
        for parameter in self.params["nominal_parameters"].keys():
            default_params[parameter] = self.params["nominal_parameters"][parameter]["default_option"]
        for parameter in self.params["constant_parameters"].keys():
            default_params[parameter] = self.params["constant_parameters"][parameter]
        return default_params

    @abstractmethod
    def reduce_dimensions(self, data, params: dict):
        pass

    def get_dimensional_reduction_out(self, data, dump_folder_path: str, reduction_params: dict,
                                      pca_preprocessing: bool = False, pca_n_components: int = 2,
                                      show_pca_diagnostic_plot: bool = False,
                                      output_buffer_param_dump_filename_extension: str = "_parameter_buffer_dump.pkl"):
        dataset_name = os.path.basename(dump_folder_path)
        dump_file_path = os.path.join(dump_folder_path,
                                      f"{self.name}" + output_buffer_param_dump_filename_extension)

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
        if dump_file_path not in self.dimensional_reduction_out_dump_data:
            # Check if the file exists
            if os.path.exists(dump_file_path):
                with open(dump_file_path, 'rb') as file:
                    self.dimensional_reduction_out_dump_data[dump_file_path] = pickle.load(file)
            else:
                # If the file doesn't exist, create it and write something to it
                with open(dump_file_path, 'wb') as file:
                    pickle.dump(buffered_data_dump, file)
                    self.dimensional_reduction_out_dump_data[dump_file_path] = buffered_data_dump

                print(f"The file '{dump_file_path}' has been created.")

        buffered_data_dump = self.dimensional_reduction_out_dump_data[dump_file_path]
        current_data = data

        if param_key not in buffered_data_dump:
            if pca_preprocessing:
                print(
                    f"Generate {self.name} with PCA preprocessing: File = {dataset_name}/{os.path.basename(dump_file_path)}, {reduction_params}, PCA n_components = {pca_n_components}")
                current_data = DimensionalReductionObject.apply_pca_preprocessing(current_data,
                                                                                  n_components=pca_n_components,
                                                                                  show_diagnostic_plot=show_pca_diagnostic_plot)
            else:
                print(
                    f"Generate {self.name}: File = {dataset_name}/{os.path.basename(dump_file_path)}, {reduction_params}")

            reduced_data = self.reduce_dimensions(current_data, **reduction_params)

            buffered_data_dump[param_key] = reduced_data

            with open(dump_file_path, 'wb') as file:
                self.dimensional_reduction_out_dump_data[dump_file_path] = buffered_data_dump
                pickle.dump(buffered_data_dump, file)

        print(
            f"Return {self.name}: File = {dataset_name}/{os.path.basename(dump_file_path)}, {reduction_params}")
        return buffered_data_dump[param_key]

    def buffer_DR_in_paramter_range(self, data, dump_folder_path, pca_preprocessing=False, pca_n_components=2):
        def iterate_over_variables(variable_names, variables_values, current_combination=[]):
            if not variables_values:
                # Base case: if no more variables, print the current combination
                # print(current_combination)
                if pca_preprocessing:
                    self.get_dimensional_reduction_out(
                        data,
                        dump_folder_path=dump_folder_path,
                        # adds the variable name back to the current combination and makes a dict to be used in the function as parameters
                        reduction_params=dict(zip(variable_names, current_combination)),
                        pca_preprocessing=True,
                        pca_n_components=pca_n_components
                    )
                else:
                    self.get_dimensional_reduction_out(
                        data,
                        dump_folder_path=dump_folder_path,
                        # adds the variable name back to the current combination and makes a dict to be used in the function as parameters
                        reduction_params=dict(zip(variable_names, current_combination)),
                        pca_preprocessing=False)
                # TODO add buffering for PCA preprocessing

            else:
                # Recursive case: iterate over the values of the current variable
                current_variable_values = variables_values[0]
                for value in current_variable_values:
                    # Recursively call the function with the next variable and the updated combination
                    iterate_over_variables(variable_names, variables_values[1:], current_combination + [value])

        # creates an array with the variable and one with all the possible values in the range
        variable_names = []
        variable_values = []
        for parameter_type in self.params.keys():
            if parameter_type == "numerical_parameters":
                for variable_name in self.params["numerical_parameters"].keys():
                    parameter_range = self.params["numerical_parameters"][variable_name].copy()
                    parameter_range.pop('value')
                    # rename key end to stop
                    parameter_range["stop"] = parameter_range.pop("end")
                    # add one step in the end to make last value of slider inclusive
                    parameter_range["stop"] = parameter_range["stop"] + parameter_range["step"]
                    values = np.arange(**parameter_range).tolist()
                    if type(parameter_range["step"]) is float:
                        # rounds values to decimal place of corresponding step variable, to avoid weird float behaviour
                        values = [round(x, get_decimal_places(parameter_range["step"])) for
                                  x in values]
                    variable_values.append(values)
                    variable_names.append(variable_name)
            elif parameter_type == "bool_parameters":
                for variable_name in self.params["bool_parameters"].keys():
                    variable_values.append([False, True])
                    variable_names.append(variable_name)

            elif parameter_type == "nominal_parameters":
                for variable_name in self.params["nominal_parameters"].keys():
                    variable_values.append(self.params["nominal_parameters"][variable_name]["options"].copy())
                    variable_names.append(variable_name)

            elif parameter_type == "constant_parameters":
                for variable_name in self.params["constant_parameters"].keys():
                    variable_values.append([self.params["constant_parameters"][variable_name]])
                    variable_names.append(variable_name)

        # generates all combinations of variable value combinations
        iterate_over_variables(variable_names, variable_values)

    def get_diagnostic_function(self, name: str):
        return self.diagnostic_functions[name]

    def list_diagnostic_functions_names(self) -> list:
        return list(self.diagnostic_functions.keys())

    def generate_config_json(self, output_file_path):
        pass

    def get_DR_parameters_dict(self):
        return self.params.copy()

    def change_params(self, params: dict = None):
        if params is None:
            self.params = self.build_in_params
        else:
            self.params = params

    def get_name(self) -> str:
        return self.name

    def __str__(self):
        return str(self.get_DR_parameters_dict())
