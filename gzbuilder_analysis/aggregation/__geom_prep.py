import numpy as np
from copy import deepcopy
from shapely.geometry import box, Point, LineString
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import scale as shapely_scale
from gzbuilder_analysis.config import DEFAULT_DISK


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
    if not comp:
        return None
    return shapely_rotate(
        shapely_scale(
            Point(comp['mux'], comp['muy']).buffer(1.0),
            xfact=comp['Re'] * comp['q'],
            yfact=comp['Re']
        ),
        comp['roll'],
        use_radians=True,
    )


def make_box(comp):
    if not comp:
        return None
    return shapely_rotate(
        box(
            comp['mux'] - comp['Re'] * comp['q'],
            comp['muy'] - comp['Re'],
            comp['mux'] + comp['Re'] * comp['q'],
            comp['muy'] + comp['Re'],
        ),
        comp['roll'],
        use_radians=True,
    )


def get_param_list(d):
    d = d or {}
    return [
        d.get(k, DEFAULT_DISK[k])
        for k in ('mux', 'muy', 'Re', 'q', 'roll')
    ]


def get_param_dict(p):
    return {
        k: v
        for k, v in zip(
            ('mux', 'muy', 'Re', 'q', 'roll'),
            p.tolist()
        )
    }


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
