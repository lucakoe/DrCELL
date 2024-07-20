from abc import abstractmethod


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