from drcell.dimensionalReduction.DimensionalReductionObject import DimensionalReductionObject


class UMAPDRObject(DimensionalReductionObject):

    def __init__(self):
        reduction_functions= {"function": util.generate_umap,
                                       "diagnostic_functions": {"diagnostic_plots": util.generate_umap_diagnostic_plot},
                                       "numeric_parameters": {
                                           "n_neighbors": {"start": 2, "end": 50, "step": 1, "value": 20},
                                           "min_dist": {"start": 0.00, "end": 1.0, "step": 0.01,
                                                        "value": 0.0}},
                                       "bool_parameters": {},
                                       "select_parameters": {},
                                       "constant_parameters": {"n_components": (2), "random_state": (42)}}


        pass

    def reduce_dimensions(self, data):
        pass
