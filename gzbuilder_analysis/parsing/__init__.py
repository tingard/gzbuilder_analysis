from copy import deepcopy
import numpy as np
from scipy.interpolate import splprep, splev
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
from gzbuilder_analysis.config import DEFAULT_SPIRAL


def downsample(points, new_n=50):
    # only keep new_n points of a spiral arm (linear interpolation)
    tck, _ = splprep(np.array(points).T, s=0)
    new_u = np.linspace(0, 1, 50)
    return np.array(splev(new_u, tck)).T


def hasDrawnComp(comp):
    return len(comp['value'][0]['value']) > 0


def parse_sersic_comp(comp, size_diff=1):
    if not hasDrawnComp(comp):
        return None
    drawing = comp['value'][0]['value'][0]
    major_axis_index = 0 if drawing['rx'] > drawing['ry'] else 1
    major_axis, minor_axis = (
        (drawing['rx'], drawing['ry'])
        if drawing['rx'] > drawing['ry']
        else (drawing['ry'], drawing['rx'])
    )
    roll = drawing['angle'] \
        + (0 if major_axis_index == 0 else 90)
    out = {
        'mu': np.array((drawing['x'], drawing['y'])) * size_diff,
        # zooniverse rotation is confusing
        'roll': np.deg2rad(roll),
        # correct for a bug in the original rendering code meaning axratio > 1
        'rEff': max(
            1e-5,
            minor_axis * float(comp['value'][1]['value']) * size_diff
        ),
        'axRatio': minor_axis / major_axis,
        'i0': float(comp['value'][2]['value']),
        'c': 2,
        'n': 1,
    }
    try:
        out['n'] = float(comp['value'][3]['value'])
        out['c'] = float(comp['value'][4]['value'])
    except IndexError:
        pass
    return out


def parse_bar_comp(comp, **kwargs):
    if not hasDrawnComp(comp):
        return None
    _comp = deepcopy(comp)
    drawing = _comp['value'][0]['value'][0]
    drawing['rx'] = drawing['width']
    drawing['ry'] = drawing['height']
    # get center position of box
    drawing['x'] = drawing['x'] + drawing['width'] / 2
    drawing['y'] = drawing['y'] + drawing['height'] / 2
    drawing['angle'] = -drawing['angle']
    _comp['value'][0]['value'][0] = drawing
    return parse_sersic_comp(_comp, **kwargs)


def parse_spiral_comp(comp, size_diff=1, image_size=256):
    out = []
    for arm in comp['value'][0]['value']:
        points = np.array(
            [[p['x'], p['y']] for p in arm['points']],
            dtype='float'
        )
        points *= size_diff
        params = {
            'i0': float(arm['details'][0]['value']),
            'spread': float(arm['details'][1]['value']),
            'falloff': max(float(comp['value'][1]['value']), 1E-5),
        }
        out.append((points, params))
    return out


def parse_annotation(annotation, size_diff=1):
    out = {'disk': None, 'bulge': None, 'bar': None, 'spiral': []}
    for component in annotation:
        if len(component['value'][0]['value']) == 0:
            # component['task'] = None
            pass
        if component['task'] == 'spiral':
            out['spiral'] = parse_spiral_comp(component, size_diff=size_diff)
        elif component['task'] == 'bar':
            out['bar'] = parse_bar_comp(component, size_diff=size_diff)
        else:
            out[component['task']] = parse_sersic_comp(
                component,
                size_diff=size_diff
            )
    return out


def parse_aggregate_model(a_m, size_diff=1.0):
    model = {}
    for c in ('disk', 'bulge', 'bar'):
        model[c] = a_m.get(c, None)
        if model[c] is not None:
            model[c]['mu'] = np.array(model[c]['mu']) * size_diff
            model[c]['rEff'] *= size_diff
    model['spiral'] = [
      [np.array(downsample(points))*size_diff, DEFAULT_SPIRAL]
      for points in a_m.get('spirals', [])
    ]
    return model


# SECTION: Model saving
def make_json(model):
    a = deepcopy(model)
    if a['disk'] is not None:
        a['disk']['mu'] = list(model['disk']['mu'])
    if a['bulge'] is not None:
        a['bulge']['mu'] = list(model['bulge']['mu'])
    if a['bar'] is not None:
        a['bar']['mu'] = list(model['bar']['mu'])
    a['spiral'] = [
        [s[0].tolist(), s[1]]
        for s in model['spiral']
    ]
    return a


def unmake_json(model):
    a = deepcopy(model)
    if a['disk'] is not None:
        a['disk']['mu'] = np.array(model['disk']['mu'])
    if a['bulge'] is not None:
        a['bulge']['mu'] = np.array(model['bulge']['mu'])
    if a['bar'] is not None:
        a['bar']['mu'] = np.array(model['bar']['mu'])
    a['spiral'] = [
        [np.array(s[0]), s[1]]
        for s in model['spiral']
    ]
    return a


# SECTION: Model sanitization
def sanitize_model(m):
    """Ensure model component  paramaters are physical
    """
    return {
        'spiral': [sanitize_spiral_param_dict(s) for s in m['spiral']],
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
    out['i0'] = max(out.get('i0', 0), 0)
    out['spread'] = abs(out.get('spread', 0))
    out['falloff'] = abs(out.get('falloff', 0))
    return points, out


def sanitize_param_dict(p):
    """Enusre physical parameters for component
    Constraints used:
        rEff > 0
        0 < axRatio < 1
        0 < roll < np.pi (not 2*pi due to rotational symmetry)
    """
    if p is None:
        return p
    # rEff > 0
    # 0 < axRatio < 1
    # 0 < roll < np.pi (not 2*pi due to rotational symmetry)
    out = deepcopy(p)
    out['rEff'] = (
        abs(out['rEff'])
        * (abs(p['axRatio']) if abs(p['axRatio']) > 1 else 1)
    )
    out['axRatio'] = min(abs(p['axRatio']), 1 / abs(p['axRatio']))
    out['roll'] = p['roll'] % np.pi
    out['i0'] = max(out.get('i0', 0), 0)
    return out


# functions for changing from pixels to arcseconds
def transform_val(v, npix, petro_theta):
        return (v - npix / 2) * 4 * petro_theta / npix


def transform_shape(shape, npix, petro_theta):
    return shapely_scale(
        shapely_translate(
            shape,
            -npix/2,
            -npix/2,
        ),
        4 * petro_theta / npix, 4 * petro_theta / npix, origin=(0, 0)
    )