import numpy as np
from copy import deepcopy
from gzbuilder_analysis.config import PARAM_BOUNDS
from .__pandas import to_pandas

def sanitize_model(m):
    """Ensure model component  paramaters are physical
    """
    return clean_model_bar({
        'spiral': list(map(sanitize_spiral_param_dict, m['spiral'])),
        **{
            k: sanitize_param_dict(v)
            for k, v in m.items()
            if k is not 'spiral'
        }
    })


def sanitize_spiral_param_dict(spiral):
    """Ensure that no values for arm parameters are less than zero
    """
    points, params = spiral
    new_params = deepcopy(params)
    new_params['I'] = max(params.get('I', 0), 0)
    new_params['spread'] = abs(params.get('spread', 0))
    new_params['falloff'] = abs(params.get('falloff', 0))
    return points, new_params


def sanitize_param_dict(params):
    """Enusre physical parameters for component
    Constraints used:
        Re > 0
        0 < q < 1
        0 < roll < np.pi (not 2*pi due to rotational symmetry)
    """
    if params is None:
        return params
    # Re > 0
    # 0 < q < 1
    # 0 < roll < np.pi (not 2*pi due to rotational symmetry)
    new_params = deepcopy(params)

    # do not allow negative values for Re, q, I, n, or c
    new_params['Re'] = abs(params.get('Re', 0))
    new_params['q'] = abs(params.get('q', 0))
    new_params['I'] = max(params.get('I', 0), PARAM_BOUNDS['I'][0])
    new_params['n'] = max(params.get('n', 0), PARAM_BOUNDS['n'][0])
    new_params['c'] = max(params.get('c', 0), PARAM_BOUNDS['c'][0])

    # if ellipticity > 1 reverse major and minor axis
    if params['q'] > 1:
        new_params['Re'] = new_params['Re'] * new_params['q']
        new_params['q'] = 1 / new_params['q']
        new_params['roll'] += np.pi / 2

    # restrict roll to be between 0 and pi (due to rotational symmetry)
    new_params['roll'] = new_params['roll'] % np.pi

    return new_params


def clean_model_bar(model):
    try:
        bar_axratio = model.get('bar', {}).get('q', 1)
        if bar_axratio > COMPONENT_CLUSTERING_PARAMS['max_bar_axratio']:
            model['bar'] = None
    except AttributeError:
        pass
    return model


def sanitize_pandas_params(params):
    params_df = params.unstack()
    for comp in ('disk', 'bulge', 'bar'):
        try:
            params_df.loc[comp] = sanitize_param_dict(params_df.loc[comp].to_dict())
        except KeyError:
            pass
    for spiral in (i for i in params_df.index.values if 'spiral' in i):
        _, new_params = sanitize_spiral_param_dict(
            ([], params_df.loc[spiral].to_dict())
        )
        params_df.loc[spiral] = new_params
    return params_df.stack().reindex(params.index)
