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
                float(comp['value'][1]['value'])
                if not ignore_scale else 1
            )
        ) / 3,
        'q': minor_axis / major_axis,
        'I': (
            float(comp['value'][2]['value'])
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
    # get center position of box
    drawing['x'] = drawing['x'] + drawing['width'] / 2
    drawing['y'] = drawing['y'] + drawing['height'] / 2
    drawing['angle'] = -drawing['angle']
    _comp['value'][0]['value'][0] = drawing
    return parse_sersic_comp(_comp, *args, **kwargs)


def parse_spiral_comp(comp, image_size, size_diff=1, **kwargs):
    out = []
    for arm in comp['value'][0]['value']:
        points = np.array(
            [
                [p['x'], image_size[1] - p['y']]
                for p in arm['points']
            ],
            dtype='float'
        )
        params = {
            # correct for 0.8 multiplier in original rendering code
            'I': float(arm['details'][0]['value']) / 0.8,
            'spread': float(arm['details'][1]['value']) * size_diff,
            'falloff': max(float(comp['value'][1]['value']), 1E-5),
        }
        out.append((points, params))
    return out


def parse_annotation(annotation, image_size, **kwargs):
    model = {'disk': None, 'bulge': None, 'bar': None, 'spiral': []}
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
        model_out['spiral'] = [
            [
                points * scale,
                {**params, 'spread': params.get('spread', np.nan) * scale}
            ]
            for points, params in model['spiral']
        ]
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
            new_spirals = [
                ((np.dot(rot_mx, (points - crpos).T).T + crpos), deepcopy(params))
                for points, params in model[comp]
            ]
            new_model[comp] = new_spirals
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
