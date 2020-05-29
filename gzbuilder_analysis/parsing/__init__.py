import json
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
from .__reproject import reproject_model
from .__sanitize import sanitize_model
from .__pandas import to_pandas, from_pandas


def __apply_to_spiral_param(func, key, spirals):
    key_ = lambda i: '{}.{}'.format(key, i)
    return {
        **spirals,
        **{
            key_(i): func(spirals[key_(i)])
            for i in range(spirals.get('n_arms', 0))
        }
    }


def __has_drawn_component(comp):
    return len(comp['value'][0]['value']) > 0


def parse_sersic_comp(comp, image_size, ignore_scale=False, **kwargs):
    if not __has_drawn_component(comp):
        return None
    drawing = comp['value'][0]['value'][0]
    major_axis = max(drawing['rx'], drawing['ry'])
    minor_axis = min(drawing['rx'], drawing['ry'])
    roll = np.deg2rad(drawing['angle'] + 90 * int(drawing['rx'] > drawing['ry']))
    out = {
        'mux': drawing['x'],
        'muy': image_size[0] - drawing['y'],
        'roll': roll,
        # in the original rendering code, minor axis was used instead of major
        # and a scaling of 3 was present
        'Re': max(
            1e-5,
            major_axis * (
                float(comp['value'][1]['value'] or 1)
                if not ignore_scale else 1
            )
        ) / 3,
        'q': minor_axis / major_axis,
        'I': (
            float(comp['value'][2]['value'] or 1)
            if comp['value'][2]['value'] is not None
            else 0.2
        ) / (2 * 0.8),  # correct for a factor of 2 and a 1/0.8 multiplier to standardise
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
    if not __has_drawn_component(comp):
        return None
    _comp = deepcopy(comp)
    drawing = _comp['value'][0]['value'][0]
    drawing['rx'] = drawing['width']
    drawing['ry'] = drawing['height']
    # get centre position of box
    drawing['x'] = drawing['x'] + drawing['width'] / 2
    drawing['y'] = drawing['y'] + drawing['height'] / 2
    drawing['angle'] = -drawing['angle']
    _comp['value'][0]['value'][0] = drawing
    return parse_sersic_comp(_comp, *args, **kwargs)


def parse_spiral_comp(comp, image_size, size_diff=1, **kwargs):
    out = {}
    i = -1
    for i, arm in enumerate(comp['value'][0]['value']):
        out.update({
            # correct for 0.8 multiplier in original rendering code
            'I.{}'.format(i): float(arm['details'][0]['value']) / 0.8,
            'spread.{}'.format(i): float(arm['details'][1]['value'] or 0.5) * size_diff,
            'falloff.{}'.format(i): max(float(comp['value'][1]['value'] or 1), 1E-5),
            'points.{}'.format(i): np.array(
                [
                    [p['x'], image_size[1] - p['y']]
                    for p in arm['points']
                ],
                dtype='float'
            )
        })
    out['n_arms'] = i + 1
    return out


def parse_annotation(annotation, image_size, **kwargs):
    model = {'disk': None, 'bulge': None, 'bar': None, 'spiral': {}}
    for component in annotation:
        if len(component['value'][0]['value']) == 0:
            # component['task'] = None
            pass
        if component['task'] == 'spiral':
            model['spiral'] = parse_spiral_comp(component, image_size)
        elif component['task'] == 'bar':
            model['bar'] = parse_bar_comp(component, image_size, **kwargs)
        else:
            model[component['task']] = parse_sersic_comp(
                component,
                image_size,
                **kwargs
            )
    # double the disk effective radius (correction systematic error)
    try:
        model['disk']['Re'] *= 2
    except TypeError:
        # this model did not have a disk
        pass
    return model


def parse_classification(classification, image_size, **kwargs):
    """Extract a more usable model from a zooniverse annotation
    """
    annotation = json.loads(classification['annotations'])
    return parse_annotation(annotation, image_size, **kwargs)


def scale_model(model, scale):
    """Scale a model to represent different image sizes
    """
    model_out = deepcopy(model)
    for comp in model.keys():
        if model[comp] is None:
            continue
        elif comp != 'spiral':
            model_out[comp]['mux'] *= scale
            model_out[comp]['muy'] *= scale
            model_out[comp]['Re'] *= scale

    model_out['spiral'] = {
        **model_out['spiral'],
        **{
            'spread.{}'.format(i): model_out['spiral'].get('spread.{}'.format(i), np.nan) * scale
            for i in range(model['spiral'].get('n_arms', 0))
        },
        **{
            'points.{}'.format(i): model_out['spiral']['points.{}'.format(i)] * scale
            for i in range(model['spiral'].get('n_arms', 0))
        },
    }
    return model_out


def rotate_model_about_centre(model, image_size, rotation):
    crpos = np.array(image_size) / 2
    rot_mx = np.array((
        (np.cos(rotation), np.sin(rotation)),
        (-np.sin(rotation), np.cos(rotation))
    ))
    new_model = {}
    for comp in model:
        if model[comp] is None:
            new_model[comp] = None
        elif comp == 'spiral':
            new_model[comp] = __apply_to_spiral_param(
                lambda s: np.dot(rot_mx, (s - crpos).T).T + crpos,
                'points', model[comp]
            )
        else:
            new_model[comp] = deepcopy(model[comp])
            p = np.array((new_model[comp]['mux'], new_model[comp]['muy']))
            new_p = np.dot(rot_mx, p - crpos) + crpos
            new_model[comp]['mux'] = new_p[0]
            new_model[comp]['muy'] = new_p[1]
            new_model[comp]['roll'] = (
                new_model[comp]['roll'] + rotation
            ) % np.pi
            # Mod Pi not 2Pi due to rotational symmetry
    return new_model


def downsample(points, new_n=50):
    """interpolate a spiral arm to reduce the number of points
    """
    tck, _ = splprep(np.array(points).T, s=0)
    new_u = np.linspace(0, 1, 50)
    return np.array(splev(new_u, tck)).T


def make_json(model):
    return json.dumps({
        **model,
        'spiral': [(points.tolist(), params) for points, params in model['spiral']]
    })


def unmake_json(f):
    model = json.load(f)
    return {
        **model,
        'spiral': [(np.array(points), params) for points, params in model['spiral']]
    }


def get_n_arms(model):
    return len()
