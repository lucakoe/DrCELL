import umap
import umap.plot
from matplotlib import pyplot as plt

import drcell.util.drCELLFileUtil
from drcell.dimensionalReduction.DimensionalReductionObject import DimensionalReductionObject


class UMAPDRObject(DimensionalReductionObject):

    def __init__(self, params: dict = None):
        self.diagnostic_functions = {"diagnostic_plots": self.generate_umap_diagnostic_plots}

        # TODO maybe check for config file here and if not there go to hardcoded values
        if params is None:
            params = {
                "numerical_parameters": {
                    "n_neighbors": {"start": 2, "end": 50, "step": 1, "value": 20},
                    "min_dist": {"start": 0.00, "end": 1.0, "step": 0.01, "value": 0.0}},
                "bool_parameters": {},
                "nominal_parameters": {},
                "constant_parameters": {
                    "n_components": (2),
                    "random_state": (42)}}
        super().__init__("UMAP", params, self.diagnostic_functions)

    def reduce_dimensions(self, data, *args, **kwargs):
        if args is None and kwargs is None:
            kwargs = self.get_default_params()
        umap_object = umap.UMAP(*args, **kwargs)
        return umap_object.fit_transform(data)

    def generate_umap_diagnostic_plots(self, data, *args, **kwargs):
        umap_object = umap.UMAP(*args, **kwargs)
        return self.generate_diagnostic_plots(umap_object, data)

    def generate_diagnostic_plots(self, umap_object, data):
        mapper = umap_object.fit(data)
        umap.plot.diagnostic(mapper, diagnostic_type='pca')
        plt.show()
        umap.plot.diagnostic(mapper, diagnostic_type='vq')
        plt.show()
        umap.plot.diagnostic(mapper, diagnostic_type='local_dim')
        plt.show()
        umap.plot.diagnostic(mapper, diagnostic_type='neighborhood')
        plt.show()
