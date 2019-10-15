import numpy as np
from copy import deepcopy
from scipy.signal import convolve2d
from numba import jit
from gzbuilder_analysis.rendering.sersic import oversampled_sersic_component
from gzbuilder_analysis.rendering.spiral import spiral_arm


# image manipulation
@jit(nopython=True)
def asinh(px):
    """Inverse hyperbolic sine function
    """
    return np.log(px + np.sqrt(1.0 + (px * px)))


@jit(nopython=True)
def asinh_stretch(px, a=0.6):
    """same as astropy.visualization.AsinhStretch, but with less faff as we
    don't care about being able to invert the transformation.
    """
    return asinh(px / a) / asinh(a)


def calculate_model(model, image_size=(256, 256), psf=None, oversample_n=5):
    """Render a model and convolve it with a psf (if provided)
    """
    disk_arr = oversampled_sersic_component(
        model['disk'],
        image_size=image_size,
        oversample_n=oversample_n)
    bulge_arr = oversampled_sersic_component(
        model['bulge'],
        image_size=image_size,
        oversample_n=oversample_n
    )
    bar_arr = oversampled_sersic_component(
        model['bar'],
        image_size=image_size,
        oversample_n=oversample_n
    )
    spirals_arr = np.add.reduce([
        spiral_arm(
            *s,
            model['disk'],
            image_size=image_size,
        )
        for s in model['spiral']
    ])
    model = disk_arr + bulge_arr + bar_arr + spirals_arr
    if psf is not None:
        return convolve2d(model, psf, mode='same', boundary='symm')
    return model


def compare_to_galaxy(arr, galaxy, psf=None, pixel_mask=None, stretch=True):
    """Given a calculated model, compare it to the galaxy data. Return a
    (optionally asinh-stretched) difference image.
    A 0.8 multiplier was present in the original rendering code
    """
    if pixel_mask is None:
        pixel_mask = np.ones(1)
    if psf is None:
        masked_model = np.nanprod((arr, pixel_mask), axis=0)
    else:
        masked_model = np.nanprod((
            convolve2d(arr, psf, mode='same', boundary='symm'),
            pixel_mask
        ), axis=0)
    masked_galaxy = np.nanprod((galaxy, pixel_mask), axis=0)
    D = (0.8 * masked_galaxy) - masked_model
    if stretch:
        return asinh_stretch(D)
    return D


def post_process(arr, psf):
    """Given a model and a psf, stretch the image to what was shown to
    volunteers
    """
    return asinh_stretch(
        convolve2d(arr, psf, mode='same', boundary='symm'),
        0.5
    )


def GZB_score(D):
    """Recreate the score shown to volunteers from an (unscaled) difference
    image
    """
    N = np.multiply.reduce(D.shape)
    return 100 * np.exp(
        -300 / N
        * np.sum(
            asinh(np.abs(D) / 0.6)**2
            / asinh(0.6)
        )
    )


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
