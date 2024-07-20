import phate

from drcell.dimensionalReduction.DimensionalReductionObject import DimensionalReductionObject


class PHATEDRObject(DimensionalReductionObject):
    def __init__(self):
        reduction_functions = {"function": generate_phate,
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


def generate_phate(data, knn=30, decay=15, t='auto', gamma=0, n_jobs=-1, n_pca=100, n_landmark=1000, verbose=False,
                   n_components=2):
    phate_operator = phate.PHATE(n_jobs=n_jobs)
    phate_operator.set_params(knn=knn, decay=decay, t=t, gamma=gamma, n_jobs=n_jobs, n_pca=n_pca, n_landmark=n_landmark,
                              verbose=verbose, n_components=n_components)

    return phate_operator.fit_transform(data)
