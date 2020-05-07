from functools import reduce
import numpy as np
from copy import deepcopy
from gzbuilder_analysis.config import COMPONENT_PARAM_BOUNDS


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

            new_params[k0] = {
                k1: np.clip(
                    new_params[k0][k1],
                    *COMPONENT_PARAM_BOUNDS[k0].get(k1, (-np.inf, np.inf))
                )
                for k1 in new_params[k0]
            }

            # restrict roll to be between 0 and pi
            # (2 degrees of rotational symmetry)
            new_params[k0]['roll'] = new_params[k0]['roll'] % np.pi
        else:
            new_params[k0] = {
                **params['spiral'],
                **reduce(lambda a, b: {**a, **b}, [
                    {
                        '{}.{}'.format(k, i): np.clip(
                            params[k0]['{}.{}'.format(k, i)],
                            *COMPONENT_PARAM_BOUNDS['spiral'][k]
                        )
                        for i in range(params['spiral']['n_arms'])
                    }
                    for k in ('I', 'spread', 'falloff')
                ])
            }
    return new_params
