import umap
import umap.plot
from matplotlib import pyplot as plt

import drcell.util.drCELLFileUtil
from drcell.dimensionalReduction.DimensionalReductionObject import DimensionalReductionObject


class UMAPDRObject(DimensionalReductionObject):

    def __init__(self):
        reduction_functions= {"function": generate_umap,
                                       "diagnostic_functions": {"diagnostic_plots": generate_umap_diagnostic_plot},
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


def generate_umap(data, n_neighbors, min_dist, n_components=2, random_state=42):
    umap_object = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    return umap_object.fit_transform(data)


def generate_umap_diagnostic_plot(data, n_neighbors, min_dist, n_components=2, random_state=42):
    umap_object = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    return generate_diagnostic_plots(umap_object, data)


def generate_diagnostic_plots(umap_object, data):
    mapper = umap_object.fit(data)
    umap.plot.diagnostic(mapper, diagnostic_type='pca')
    plt.show()
    umap.plot.diagnostic(mapper, diagnostic_type='vq')
    plt.show()
    umap.plot.diagnostic(mapper, diagnostic_type='local_dim')
    plt.show()
    umap.plot.diagnostic(mapper, diagnostic_type='neighborhood')
    plt.show()
