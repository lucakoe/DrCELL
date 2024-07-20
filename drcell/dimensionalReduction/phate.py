from drcell.dimensionalReduction.DimensionalReductionObject import DimensionalReductionObject


class PHATEDRObject(DimensionalReductionObject):
    def __init__(self):
        reduction_functions = {"function": util.generate_phate,
                                        "diagnostic_functions": {},
                                        "numeric_parameters": {"knn": {"start": 5, "end": 100, "step": 1, "value": 30},
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
                                        "select_parameters": {},
                                        "constant_parameters": {"n_components": (2), "n_jobs": (-1)}}
