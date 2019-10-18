import numpy as np
from copy import deepcopy

def sanitize_model(m):
    """Ensure model component  paramaters are physical
    """
    return {
        'spiral': sanitize_spiral_param_dict(m['spiral']),
        **{
            k: sanitize_param_dict(v)
            for k, v in m.items()
            if k is not 'spiral'
        }
    }


def sanitize_spiral_param_dict(p):
    """Ensure that no values for arm parameters are less than zero
    """
    points, params = p
    out = deepcopy(params)
    out['I'] = max(out.get('I', 0), 0)
    out['spread'] = abs(out.get('spread', 0))
    out['falloff'] = abs(out.get('falloff', 0))
    return points, out


def sanitize_param_dict(p):
    """Enusre physical parameters for component
    Constraints used:
        Re > 0
        0 < q < 1
        0 < roll < np.pi (not 2*pi due to rotational symmetry)
    """
    if p is None:
        return p
    # Re > 0
    # 0 < q < 1
    # 0 < roll < np.pi (not 2*pi due to rotational symmetry)
    out = deepcopy(p)
    out['Re'] = (
        abs(out['Re'])
        * (abs(p['q']) if abs(p['q']) > 1 else 1)
    )
    out['q'] = min(abs(p['q']), 1 / abs(p['q']))
    out['roll'] = p['roll'] % np.pi
    out['I'] = max(out.get('I', 0), 0)
    return out
