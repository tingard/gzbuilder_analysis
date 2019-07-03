import numpy as np

# Default parameters for model components
DEFAULT_DISK = {
    'mu': np.zeros(2) + 50,
    'roll': 0.0,
    'rEff': 100.0,
    'axRatio': 1.0,
    'c': 2.0,
}

DEFAULT_SPIRAL = {
    'i0': 0.1, 'spread': 0.5, 'falloff': 1.0,
}

SPIRAL_LOF_KWARGS = {
    'n_neighbors': 20,
    'contamination': 'auto',
    'novelty': True
}

# Hyper-priors on bayesian ridge regression for log spiral fitting
SPIRAL_BAYESIAN_RIDGE_PRIORS = {
    'alpha_1': 1.0230488753392505e-08,
    'alpha_2': 0.6923902410146074,
}

# DBSCAN parameters for spiral arm clustering
ARM_CLUSTERING_PARAMS = {
    'eps': 400,
    'min_samples': 4,
}

# DBSCAN parameters for spiral arm clustering
COMPONENT_CLUSTERING_PARAMS = {
    'disk': {'eps': 0.3, 'min_samples': 5},
    'bulge': {'eps': 0.3, 'min_samples': 3},
    'bar': {'eps': 0.3846467, 'min_samples': 3},
    'max_bar_axratio': 0.6,
}


# Defaults for fitting
FIT_PARAMS = {
    'disk':   ('i0', 'rEff', 'axRatio'),
    'bulge':  ('i0', 'rEff', 'axRatio', 'n'),
    'bar':    ('i0', 'rEff', 'axRatio', 'n', 'c'),
    'spiral': ('i0', 'spread', 'falloff'),
}

PARAM_BOUNDS = {
    'i0': (0, 50),
    'rEff': (0, 1E4),
    'axRatio': (0.2, 1),
    'n': (0.3, 10),
    'c': (1E-2, 1E1),
    'spread': (0, 1E2),
    'falloff': (1E-2, 1E10),
}
