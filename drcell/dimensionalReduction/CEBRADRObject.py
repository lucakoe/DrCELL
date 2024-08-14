import cebra

from drcell.dimensionalReduction.DimensionalReductionObject import DimensionalReductionObject


class CEBRADRObject(DimensionalReductionObject):
    def __init__(self, params: dict = None):
        self.diagnostic_functions = {}
        if params is None:
            params = {
                "numerical_parameters": {"max_iterations": {"start": 1000, "end": 100000, "step": 100,
                                                            "value": 10000},
                                         "learning_rate [1e]": {"start": -5, "end": -2, "step": 1,
                                                                "value": -3},
                                         "batch_size": {"start": 32, "end": 512, "step": 1,
                                                        "value": 128},

                                         },

                "bool_parameters": {},
                "nominal_parameters": {
                },
                "constant_parameters": {"output_dimension": (2)}}

        super().__init__("CEBRA", params)

    def reduce_dimensions(self, data, *args, **kwargs):
        kwargs["learning_rate"] = 10 ** kwargs["learning_rate [1e]"]
        kwargs.pop("learning_rate [1e]")
        model = cebra.CEBRA(*args, **kwargs)
        model.fit(data)
        reduced_data = model.transform(data)
        # the format of the output follows the sklearn .fit_transform function
        return reduced_data
