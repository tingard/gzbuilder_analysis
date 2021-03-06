import numpy as np

# Default parameters for model components
DEFAULT_DISK = {
    'mux': 50.0,
    'muy': 50.0,
    'roll': 0.0,
    'Re': 100.0,
    'q': 1.0,
    'c': 2.0,
    'I': 1.0,
}

DEFAULT_SPIRAL = {
    'I': 0.1, 'spread': 3, 'falloff': 1.0,
}

# Parameters for Local Outlier Factor cleaning of spiral arm clusters
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
COMPONENT_CLUSTERING_PARAMS = {
    'disk': {'eps': 0.3, 'min_samples': 4},
    'bulge': {'eps': 0.4, 'min_samples': 4},
    'bar': {'eps': 0.47839828189143974, 'min_samples': 4},
    'spiral': {'eps': 1E-3, 'min_samples': 4, 'merging_distance': 5E-4},
    'max_bar_axratio': 0.6,
}

# Defaults for fitting
ALL_PARAMS = {
    'disk':   ('mux', 'muy', 'roll', 'Re', 'q', 'I', 'n', 'c'),
    'bulge':  ('mux', 'muy', 'roll', 'Re', 'q', 'I', 'n', 'c'),
    'bar':    ('mux', 'muy', 'roll', 'Re', 'q', 'I', 'n', 'c'),
    'spiral': ('I', 'spread', 'falloff'),
}


COMPONENT_PARAM_BOUNDS = {
    'disk': {
        'I': [0.0, np.inf],
        'L': [0.0, np.inf],
        'mux': [-np.inf, np.inf],
        'muy': [-np.inf, np.inf],
        'q': [0.25, 1.2],
        'roll': [-np.inf, np.inf],
        'Re': [0.01, np.inf],
        'n': [1, 1],
    },
    'bulge': {
        'I': [0.0, np.inf],
        'Re': [0.0, np.inf],
        'frac': [0.0, 0.99],
        'mux': [-np.inf, np.inf],
        'muy': [-np.inf, np.inf],
        'n': [0.5, 5],
        'c': [2, 2],
        'q': [0.6, 1.2],
        'roll': [-np.inf, np.inf],
        'scale': [0.05, 1],
    },
    'bar': {
        'I': [0.0, np.inf],
        'Re': [0.0, np.inf],
        'c': [1, 6],
        'frac': [0.0, 0.99],
        'mux': [-np.inf, np.inf],
        'muy': [-np.inf, np.inf],
        'n': [0.3, 5],
        'q': [0.05, 0.5],
        'roll': [-np.inf, np.inf],
        'scale': [0.05, 1],
    },
    'spiral': {
        'I': [0, np.inf],
        'A': [0, np.inf],
        'falloff': [0.001, np.inf],
        'phi': [-85.0, 85.0],
        'spread': [0.5, 15],
        't_min': [-np.inf, np.inf],
        't_max': [-np.inf, np.inf],
        'n_arms': [0, np.inf],
    },
    'centre': {
        'mux': [-np.inf, np.inf],
        'muy': [-np.inf, np.inf],
    }
}
