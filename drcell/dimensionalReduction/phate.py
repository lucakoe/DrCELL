import phate

from drcell.dimensionalReduction.DimensionalReductionObject import DimensionalReductionObject


class PHATEDRObject(DimensionalReductionObject):
    def __init__(self, params: dict = None):
        # TODO maybe check for config file here and if not there go to hardcoded values
        if params is None:
            params = {
                "numerical_parameters": {"knn": {"start": 5, "end": 100, "step": 1, "value": 30},
                                       "decay": {"start": 1, "end": 50, "step": 1,
                                                 "value": 15},
                                       "t": {"start": 5, "end": 100, "step": 1, "value": 5},
                                       "gamma": {"start": 0, "end": 10, "step": 0.1,
                                                 "value": 0},

                                       "n_pca": {"start": 5, "end": 100, "step": 1,
                                                 "value": 100},
                                       "n_landmark": {"start": 50, "end": 1000, "step": 10,
                                                      "value": 1000},

                                       },
                "bool_parameters": {"verbose": False},
                "nominal_parameters": {},
                "constant_parameters": {"n_components": (2), "n_jobs": (-1)}}
        super().__init__("PHATE", params)

    def reduce_dimensions(self, data, *args, **kwargs):
        if args is None and kwargs is None:
            kwargs = self.get_default_params()
        phate_operator = phate.PHATE()
        phate_operator.set_params(*args, **kwargs)
        return phate_operator.fit_transform(data)


def generate_phate(data,arg):
    phate_operator = phate.PHATE()
    phate_operator.set_params(**arg)

    return phate_operator.fit_transform(data)
