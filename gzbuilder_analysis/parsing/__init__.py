from copy import deepcopy
import numpy as np
from scipy.interpolate import splprep, splev
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
from gzbuilder_analysis.config import DEFAULT_SPIRAL
from gzbuilder_analysis.spirals.deprojecting import change_wcs

def downsample(points, new_n=50):
    # only keep new_n points of a spiral arm (linear interpolation)
    tck, _ = splprep(np.array(points).T, s=0)
    new_u = np.linspace(0, 1, 50)
    return np.array(splev(new_u, tck)).T


def hasDrawnComp(comp):
    return len(comp['value'][0]['value']) > 0


def parse_sersic_comp(comp, image_size, size_diff=1):
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
        'mux': drawing['x'] * size_diff,
        'muy': image_size[0] - drawing['y'] * size_diff,
        # zooniverse rotation is confusing
        'roll': -np.deg2rad(roll),
        # correct for a bug in the original rendering code meaning axratio > 1
        'rEff': max(
            1e-5,
            minor_axis * float(comp['value'][1]['value']) * size_diff
        ),
        'axRatio': minor_axis / major_axis,
        'i0': (
            float(comp['value'][2]['value'])
            if comp['value'][2]['value'] is not None
            else 0.2
        ),
        'c': 2,
        'n': 1,
    }
    try:
        out['n'] = float(comp['value'][3]['value'])
        out['c'] = float(comp['value'][4]['value'])
    except IndexError:
        pass
    return out


def parse_bar_comp(comp, *args, **kwargs):
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
    return parse_sersic_comp(_comp, *args, **kwargs)


def parse_spiral_comp(comp, image_size, size_diff=1):
    out = []
    for arm in comp['value'][0]['value']:
        points = np.array(
            [
                [p['x'] * size_diff, image_size[1] - p['y']*size_diff]
                for p in arm['points']
            ],
            dtype='float'
        )
        params = {
            'i0': float(arm['details'][0]['value']),
            'spread': float(arm['details'][1]['value']),
            'falloff': max(float(comp['value'][1]['value']), 1E-5),
        }
        out.append((points, params))
    return out


def __rotation_matrix(r):
#     r = np.deg2rad(d)
    return np.array((
        (np.cos(r), np.sin(r)),
        (-np.sin(r), np.cos(r))
    ))


def correct_sersic_comp(comp, wcs_in, wcs_out):
    centre = np.array((comp['mux'], comp['muy']))
    major_axis = (
        centre
        + np.dot(__rotation_matrix(comp['roll']), (comp['rEff'], 0))
    )
    minor_axis = (
        centre
        + np.dot(
            __rotation_matrix(comp['roll'] + np.pi / 2),
            (comp['rEff'] * comp['axRatio'], 0)
        )
    )
    world_points = wcs_in.all_pix2world([
        centre, major_axis, minor_axis
    ], 0)
    (
        new_centre, new_major_axis, new_minor_axis
    ) = wcs_out.all_world2pix(world_points, 0)
    comp_out = {
        **comp,
        'mux': new_centre[0],
        'muy': new_centre[1],
        'roll': -np.arctan2(*(new_major_axis - new_centre)[::-1]),
        'rEff': np.sqrt(np.add.reduce((new_major_axis - new_centre)**2)),
    }
    comp_out['axRatio'] = np.sqrt(
        np.add.reduce((new_minor_axis - new_centre)**2)
    ) / comp_out['rEff']
    return comp_out


def correct_model_using_wcs(model_in, wcs_in, wcs_out):
    model_out = deepcopy(model_in)
    for k in model_out.keys():
        if model_out[k] is None:
            continue
        if k == 'spiral':
            model_out[k] = correct_spirals(model_out[k], wcs_in, wcs_out)
        else:
            model_out[k] = correct_sersic_comp(model_out[k], wcs_in, wcs_out)
    return model_out


def correct_spirals(spirals, wcs_in, wcs_out):
    return [
        [change_wcs(spiral[0], wcs_in, wcs_out), spiral[1]]
        for spiral in spirals
    ]


def parse_annotation(annotation, image_size, size_diff=1, wcs_in=None, wcs_out=None):
    model = {'disk': None, 'bulge': None, 'bar': None, 'spiral': []}
    for component in annotation:
        if len(component['value'][0]['value']) == 0:
            # component['task'] = None
            pass
        if component['task'] == 'spiral':
            model['spiral'] = parse_spiral_comp(component, image_size, size_diff)
        elif component['task'] == 'bar':
            model['bar'] = parse_bar_comp(component, image_size, size_diff)
        else:
            model[component['task']] = parse_sersic_comp(
                component,
                image_size,
                size_diff=size_diff
            )
    if wcs_in is not None and wcs_out is not None:
        return correct_model_using_wcs(model, wcs_in, wcs_out)
    return model


def scale_aggregate_model(a_m, image_size, size_diff=1.0,
                          wcs_in=None, wcs_out=None):
    model = {}
    for c in ('disk', 'bulge', 'bar'):
        model[c] = a_m.get(c, None)
        if model[c] is not None:
            model[c]['mux'] *= size_diff
            model[c]['muy'] = (model[c]['muy'] * size_diff) - image_size[0]
            model[c]['rEff'] *= size_diff
    model['spiral'] = [
        [np.array(downsample(points))*size_diff, DEFAULT_SPIRAL]
        for points in a_m.get('spirals', [])
    ]
    model['spiral'] = [
        [(0, image_size[1]) - points, params]
        for points, params in model['spiral']
    ]
    if wcs_in is not None and wcs_out is not None:
        return correct_model_using_wcs(model, wcs_in, wcs_out)
    return model


def scale_model_errors(e, size_diff=1.0):
    error = {}
    for c in ('disk', 'bulge', 'bar'):
        error[c] = deepcopy(e.get(c, None))
        if e.get(c, None) is not None:
            error[c]['rEff'] *= size_diff
    return error


def standardise_model(model):
    """Galaxy Buildler models do not take the form
    $\Sigma(r ) = \Sigma_e\exp\left[-\kappa\left(\left(\frac{r}{r_e}\right)^{1/n} - 1\right)\right]$
    Instead, $\Sigma_e = \frac{i_0}{2};\; \kappa = b_n;\; D = \left(\frac{r}{r_e}\right)$
    So we must transform parameters accordingly
    """
    standardised_model = deepcopy(model)
    for k in ('disk', 'bulge', 'bar'):
        standardised_model[k]['Ie'] = standardised_model[k]['i0'] / 2
        standardised_model[k]['Re'] = standardised_model[k]['rEff'] / 3
    return standardised_model


# SECTION: Model saving
def make_json(model):
    try:
        a = deepcopy(model)
        if a.get('spiral', None) is not None:
            a['spiral'] = [
                [s[0].tolist(), s[1]]
                for s in model['spiral']
            ]
        return a
    except TypeError:
        return np.nan


def unmake_json(model):
    for k in ('disk', 'bulge', 'bar'):
        if model[k] is not None and 'mu' in model[k]:
            mu = model[k].pop('mu')
            model[k]['mux'] = mu[0]
            model[k]['muy'] = mu[1]
    a = deepcopy(model)
    a['spiral'] = [
        [np.array(s[0]), s[1]]
        for s in model['spiral']
    ]
    return a


# SECTION: Model scaling
def make_unscaled_model(scaled_model, multiplier=1.0):
    """recover from the multiplier used for rendering, available in the diff
    data. Also removes the 1/0.8 scaling used
    """
    model = deepcopy(scaled_model)
    for k in ('disk', 'bulge', 'bar'):
        try:
            model[k]['i0'] *= multiplier / 0.8
        except TypeError:
            pass
    try:
        for i in range(len(model['spiral'])):
            model['spiral'][i][1]['i0'] *= multiplier / 0.8
    except TypeError as e:
        print(len(model['spiral']))
        print(
            [model['spiral'][i][1]['i0'] for i in range(len(model['spiral']))],
            multiplier
        )
        raise(e)
    return model


# SECTION: Model sanitization
def sanitize_model(m):
    """Ensure model component  paramaters are physical
    """
    return {
        'spiral': [sanitize_spiral_param_dict(s) for s in m['spiral']],
        **{
            k: sanitize_param_dict(v)
            for k, v in m.items()
            if k != 'spiral'
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
        0 < roll < np.pi (not 2*pi due to rotational symmetry of boxy ellipses)
    """
    if p is None:
        return p
    # rEff > 0
    # 0 < axRatio < 1
    # 0 < roll < np.pi (not 2*pi due to rotational symmetry)
    out = deepcopy(p)
    out['rEff'] = (
        abs(p['rEff'])
        * (abs(p['axRatio']) if abs(p['axRatio']) > 1 else 1)
    )
    out['axRatio'] = min(abs(p['axRatio']), 1 / abs(p['axRatio']))
    out['roll'] = p['roll'] % np.pi
    out['i0'] = max(p.get('i0', 0), 0)
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
