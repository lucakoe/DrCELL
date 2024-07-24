from drcell import DimensionalReductionObject


class CustomDimensionalReductionObjectExample(DimensionalReductionObject):
    def __init__(self, params: dict = None):
        self.diagnostic_functions = {"custom_diagnostic_function_1": self.custom_diagnostic_function_1,
                                     "custom_diagnostic_function_2": self.custom_diagnostic_function_2,
                                     "custom_diagnostic_function_n": self.custom_diagnostic_function_n}
        if params is None:
            params = {
                "numeric_parameters": {"numeric_example_parameter_1": {"start": 5, "end": 50, "step": 1, "value": 30},
                                       "numeric_example_parameter_2": {"start": 10, "end": 200, "step": 10,
                                                                       "value": 200},
                                       "numeric_example_parameter_n": {"start": 250, "end": 1000, "step": 10,
                                                                       "value": 1000},
                                       },
                "bool_parameters": {"bool_example_parameter_1": False, "bool_example_parameter_2": True,
                                    "bool_example_parameter_n": False},
                "nominal_parameters": {
                    "nominal_example_parameter_1": {
                        "options": ["example_option_1", "example_option_2", "example_option_n"],
                        "default_option": "example_option_2"},
                    "nominal_example_parameter_2": {
                        "options": ["example_option_1", "example_option_2", "example_option_n"],
                        "default_option": "example_option_1"},
                    "nominal_example_parameter_n": {
                        "options": ["example_option_1", "example_option_2", "example_option_n"],
                        "default_option": "example_option_n"}
                },
                "constant_parameters": {"n_components": (2),
                                        "constant_parameter_1": ("example_parameter"),
                                        "constant_parameter_2": (5),
                                        "constant_parameter_n": (True)}}

        super().__init__("CustomDRFunction", params)

    def reduce_dimensions(self, data, params: dict = None):
        # put dimensional reduction function here!!!
        # the format of the output follows the sklearn .fit_transform function
        return None

    def custom_diagnostic_function_1(self, data, *args, **kwargs) -> None:
        # DrCELL will call this function with the current data and all the current parameters of
        # the dimensional reduction function as args.

        # some diagnostic function
        pass

    def custom_diagnostic_function_2(self, data, *args, **kwargs) -> None:
        # DrCELL will call this function with the current data and all the current parameters of
        # the dimensional reduction function as args.

        # some diagnostic function
        pass

    def custom_diagnostic_function_n(self, data, *args, **kwargs) -> None:
        # DrCELL will call this function with the current data and all the current parameters of
        # the dimensional reduction function as args.

        # some diagnostic function
        pass
