import sklearn.manifold

from drcell.dimensionalReduction.DimensionalReductionObject import DimensionalReductionObject


class TSNEDRObject(DimensionalReductionObject):
    def __init__(self):
        reduction_functions = {"function": generate_t_sne,
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


def generate_t_sne(data, perplexity=30, learning_rate=200, n_iter=1000, early_exaggeration=12, angle=0.5,
                   metric="euclidean", n_components=2):
    tsne_operator = sklearn.manifold.TSNE(perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter,
                                          early_exaggeration=early_exaggeration, angle=angle,
                                          metric=metric, n_components=n_components)
    return tsne_operator.fit_transform(data)
