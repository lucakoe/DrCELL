from drcell.dimensionalReduction.DimensionalReductionObject import DimensionalReductionObject


class TSNEDRObject(DimensionalReductionObject):
    def __init__(self):
        reduction_functions = {"function": util.generate_t_sne,
                                        "diagnostic_functions": {},
                                        "numeric_parameters": {
                                            "perplexity": {"start": 5, "end": 50, "step": 1, "value": 30},
                                            "learning_rate": {"start": 10, "end": 200, "step": 10,
                                                              "value": 200},
                                            "n_iter": {"start": 250, "end": 1000, "step": 10, "value": 1000},
                                            "early_exaggeration": {"start": 4, "end": 20, "step": 1,
                                                                   "value": 12},
                                            "angle": {"start": 0.2, "end": 0.8, "step": 0.1, "value": 0.5}},
                                        "bool_parameters": {},
                                        "select_parameters": {
                                            "metric": {"options": ["euclidean", "manhattan", "cosine"],
                                                       "default_option": "euclidean"}},
                                        "constant_parameters": {"n_components": (2)}}




