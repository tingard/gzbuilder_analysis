import numpy as np
from copy import deepcopy


def __rotation_matrix(r):
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
    if comp is None:
        return None
    centre = np.array((comp['mux'], comp['muy']))
    major_vec = np.dot(__rotation_matrix(-comp['roll']), (0, comp['Re']))
    minor_vec = np.dot(__rotation_matrix(-comp['roll']), (comp['Re'] * comp['q'], 0))
    major_axis = np.stack((centre, major_vec + centre), axis=1)
    minor_axis = np.stack((centre, minor_vec + centre), axis=1)
    new_centre = np.array(
        wcs_out.all_world2pix(*wcs_in.all_pix2world(*centre, 0), 0)
    )
    new_major_axis = np.array(
        wcs_out.all_world2pix(*wcs_in.all_pix2world(*major_axis, 0), 0)
    )
    new_minor_axis = np.array(
        wcs_out.all_world2pix(*wcs_in.all_pix2world(*minor_axis, 0), 0)
    )
    major_ax_vec = new_major_axis[:, 1] - new_centre
    minor_ax_vec = new_minor_axis[:, 1] - new_centre
    comp_out = {
        **comp,
        'mux': new_centre[0],
        'muy': new_centre[1],
        'roll': -np.arctan2(major_ax_vec[0], major_ax_vec[1]),
        'Re': np.linalg.norm(major_ax_vec)
    }
    comp_out['q'] = np.linalg.norm(minor_ax_vec) / comp_out['Re']
    return comp_out


def reproject_spirals(spirals, wcs_in, wcs_out):
    return {**spirals, **{
        'points.{}'.format(i): change_spiral_wcs(spirals['points.{}'.format(i)], wcs_in, wcs_out)
        for i in range(spirals.get('n_arms', 0))
    }}


def reproject_model(model, wcs_in, wcs_out):
    """Rotate, translate and scale model from one WCS to another
    """
    model_out = deepcopy(model)
    return dict(
        disk=reproject_sersic(model_out['disk'], wcs_in, wcs_out),
        bulge=reproject_sersic(model_out['bulge'], wcs_in, wcs_out),
        bar=reproject_sersic(model_out['bar'], wcs_in, wcs_out),
        spiral=reproject_spirals(model_out['spiral'], wcs_in, wcs_out)
    )
