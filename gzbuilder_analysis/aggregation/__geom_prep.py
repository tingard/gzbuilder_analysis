import numpy as np
from copy import deepcopy
from shapely.geometry import box, Point
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
from gzbuilder_analysis.config import DEFAULT_DISK
from ..parsing.__sanitize import sanitize_param_dict

def reset_task_scale_slider(task):
    _t = deepcopy(task)
    _t['value'][1]['value'] = 1.0
    return _t


def remove_scaling(annotation):
    return [
        reset_task_scale_slider(task)
        for task in annotation
    ]


def make_ellipse(comp):
    return shapely_rotate(
        shapely_scale(
            Point(comp['mux'], comp['muy']).buffer(1.0),
            xfact=comp['Re'] * comp['q'],
            yfact=comp['Re']
        ),
        np.rad2deg(comp['roll'])
    ) if comp else None


def make_box(comp):
    if not comp or comp['q'] == 0:
        return None
    return shapely_rotate(
        box(
            comp['mux'] - comp['Re'] / 2,
            comp['muy'] - comp['Re'] / 2 / comp['q'],
            comp['mux'] + comp['Re'] / 2,
            comp['muy'] + comp['Re'] / 2 / comp['q'],
        ),
        np.rad2deg(comp['roll'])
    )


def get_param_list(d):
    d = d or {}
    return [
        d.get(k, DEFAULT_DISK[k])
        for k in ('mux', 'muy', 'Re', 'q', 'roll')
    ]


def get_param_dict(p):
    return sanitize_param_dict({
        k: v
        for k, v in zip(
            ('mux', 'muy', 'Re', 'q', 'roll'),
            p.tolist()
        )
    })


def ellipse_from_param_list(p):
    return make_ellipse(get_param_dict(p))


def box_from_param_list(p):
    return make_box(get_param_dict(p))


def get_drawn_arms(models, clean=True, image_size=None):
    """Given classifications for a galaxy, get the non-self-overlapping with
    more then five points drawn spirals arms
    """
    arms = np.array([
        points
        for model in models.values
        for points, params in model['spiral']
        if clean and (len(points) > 5 and LineString(arm).is_simple)
    ])
    if image_size is not None:
        # reverse the y-axis
        return np.array([
            (1, -1) * (arm - (0, image_size[0]))
            for arm in arms
        ])
    return arms
