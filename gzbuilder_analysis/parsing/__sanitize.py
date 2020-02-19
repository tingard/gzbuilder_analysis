import numpy as np
from copy import deepcopy
from gzbuilder_analysis.config import COMPONENT_CLUSTERING_PARAMS, \
    COMPONENT_PARAM_BOUNDS


def sanitize_model(params):
    """Enusre physical parameters for component
    """
    new_params = deepcopy(params)

    # enforce constraints on models
    for k0 in new_params:
        if new_params[k0] is None:
            continue
        if k0 != 'spiral':
            # if ellipticity > 1 reverse major and minor axis
            if new_params[k0]['q'] > 1:
                new_params[k0]['Re'] = new_params[k0]['Re'] * new_params[k0]['q']
                new_params[k0]['q'] = 1 / new_params[k0]['q']
                new_params[k0]['roll'] += np.pi / 2
            for k1 in new_params[k0]:
                new_params[k0][k1] = np.clip(
                    new_params[k0][k1],
                    *COMPONENT_PARAM_BOUNDS[k0].get(k1, (-np.inf, np.inf))
                )
            # restrict roll to be between 0 and pi
            # (2 degrees of rotational symmetry)
            new_params[k0]['roll'] = new_params[k0]['roll'] % np.pi
        else:
            if type(new_params[k0]) in (list, tuple):
                for i in range(len(new_params[k0])):
                    for k1 in new_params[k0][i][1]:
                        new_params[k0][i][1][k1] = np.clip(
                            new_params[k0][i][1][k1],
                            *COMPONENT_PARAM_BOUNDS[k0].get(k1, (-np.inf, np.inf))
                        )
    return new_params
