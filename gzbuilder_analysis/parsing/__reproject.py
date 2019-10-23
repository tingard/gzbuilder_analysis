import numpy as np
from copy import deepcopy
# model reprojection

def __rotation_matrix(r):
#     r = np.deg2rad(d)
    return np.array((
        (np.cos(r), np.sin(r)),
        (-np.sin(r), np.cos(r))
    ))


def change_spiral_wcs(points, wcs_in, wcs_out):
    return wcs_out.all_world2pix(
        wcs_in.all_pix2world(points, 0),
        0
    )


def reproject_sersic(comp, wcs_in, wcs_out):
    centre = np.array((comp['mux'], comp['muy']))
    major_axis = (
        centre
        + np.dot(__rotation_matrix(comp['roll']), (comp['Re'], 0))
    )
    minor_axis = (
        centre
        + np.dot(
            __rotation_matrix(comp['roll'] + np.pi / 2),
            (comp['Re'] * comp['q'], 0)
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
        'Re': np.sqrt(np.add.reduce((new_major_axis - new_centre)**2)),
    }
    comp_out['q'] = np.sqrt(
        np.add.reduce((new_minor_axis - new_centre)**2)
    ) / comp_out['Re']
    return comp_out


def reproject_spirals(spirals, wcs_in, wcs_out):
    return [
        [change_spiral_wcs(spiral[0], wcs_in, wcs_out), spiral[1]]
        for spiral in spirals
    ]


def reproject_model(model, wcs_in, wcs_out):
    """Rotate, translate and scale model from one WCS to another
    """
    model_out = deepcopy(model)
    for k in model_out.keys():
        if model.get(k, None) is None:
            continue
        if k == 'spiral':
            model_out[k] = reproject_spirals(model[k], wcs_in, wcs_out)
        else:
            model_out[k] = reproject_sersic(model[k], wcs_in, wcs_out)
    return model_out
