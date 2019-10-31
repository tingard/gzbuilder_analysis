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
    'I': 0.1, 'spread': 0.5, 'falloff': 1.0,
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
ALL_PARAMS = {
    'disk':   ('mux', 'muy', 'roll', 'Re', 'q', 'I', 'n', 'c'),
    'bulge':  ('mux', 'muy', 'roll', 'Re', 'q', 'I', 'n', 'c'),
    'bar':    ('mux', 'muy', 'roll', 'Re', 'q', 'I', 'n', 'c'),
    'spiral': ('I', 'spread', 'falloff'),
}

FIT_PARAMS = {
    'disk':   ('I', 'Re', 'q'),
    'bulge':  ('I', 'Re', 'q', 'n'),
    'bar':    ('I', 'Re', 'q', 'n', 'c'),
    'spiral': ('I', 'spread', 'falloff'),
}

SLIDER_FIT_PARAMS = {
    'disk': ('I',),
    'bulge': ('I', 'n'),
    'bar': ('I', 'n', 'c'),
    'spiral': ('I', 'spread', 'falloff'),
}


PARAM_BOUNDS = {
    'I': (0, np.inf),
    'mux': (-np.inf, np.inf),
    'muy': (-np.inf, np.inf),
    'Re': (0, np.inf),
    'q': (1E-2, 1E2),
    'n': (2E-1, 10),
    'c': (1E-1, 1E1),
    'spread': (0, np.inf),
    'falloff': (1E-2, np.inf),
    'roll': (-np.inf, np.inf)
}


# dict of parameter limits and default values
BAD_PARAM_VALUES = {
    'disk': {
        # inf not 1 as q of 1 is okay
        'q': (0, 0.5, np.inf),
        'scale': (0, 2, 1),
        'I': (0, 1, 0.2),
    },
    'bulge': {
        'q': (0, 0.5, np.inf),
        'scale': (0, 2, 1),
        'I': (0, 2, 0.5),
        'n': (0.5, 5, 1),
    },
    'bar': {
        'roll': (-np.inf, 0.0, np.inf),
        'scale': (0, 2, 1),
        'I': (0, 1, 0.2),
        'n': (0.3, 2, 0.5),
        'c': (1.5, 3, 2),
    },
    'spiral': {
        'I': (0, 1, 0.75),
        'spread': (0, 2, 1),
        'falloff': (0, 2, 1),
    }
}
