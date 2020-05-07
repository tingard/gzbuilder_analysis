import jax.numpy as np
from jax import vmap
from jax import jit
from jax.config import config
from gzbuilder_analysis.config import DEFAULT_DISK, DEFAULT_SPIRAL
from .sersic import sersic
config.update("jax_enable_x64", True)


def line_segment_distance(a, cx, cy):
    px0, py0, px1, py1 = a
    ux = cx - px0
    uy = cy - py0
    vx = px1 - px0
    vy = py1 - py0
    dot = ux * vx + uy * vy
    t = np.clip(dot / (vx**2 + vy**2), 0, 1)
    return np.sqrt((vx*t - ux)**2 + (vy*t - uy)**2)


def vmap_polyline_distance(polyline, cx, cy):
    p = np.concatenate((polyline[:-1], polyline[1:]), axis=-1)
    return np.min(vmap(line_segment_distance, (0, None, None))(p, cx, cy), axis=0)


@jit
def __rotmx(a):
    return np.array(((np.cos(a), np.sin(a)), (-np.sin(a), np.cos(a))))


@jit
def __lsp(A, phi, theta):
    return (
        A*np.exp(theta * np.tan(np.deg2rad(phi)))
        * np.stack((np.cos(theta), np.sin(theta)))
    ).T


@jit
def inclined_lsp(A, phi, q, psi, theta):
    Q = np.array(((q, 0), (0, 1)))
    elliptcial = np.squeeze(
        np.dot(Q, np.expand_dims(__lsp(A, phi, theta), -1))
    ).T
    return np.squeeze(np.dot(__rotmx(-psi), np.expand_dims(elliptcial, -1))).T


@jit
def correct_logsp_params(A, phi, q, psi, dpsi, theta):
    Ap = np.exp(-dpsi * np.tan(np.deg2rad(phi)))
    return A * Ap, phi, q, psi, theta + dpsi


@jit
def corrected_inclined_lsp(A, phi, q, psi, dpsi, theta):
    return inclined_lsp(
        *correct_logsp_params(A, phi, q, psi, dpsi, theta)
    )


@jit
def translate_spiral(lsp, mux, muy):
    return lsp + np.array((mux, muy))


def spiral_arm(arm_points=None, distances=None, params=DEFAULT_SPIRAL,
               disk=DEFAULT_DISK, image_size=(256, 256)):
    """Helper function to calculate what spiral arms look like for a given disk
    Not recommended as `renderer.render_comps is faster`
    """
    if arm_points is None and distances is None:
        raise TypeError(
            'Must provide either an (N,2) array of points,'
            'or an (N, M) distance matrix'
        )
    # catch some easy-to-calculate cases
    if (
        # we don't have a disk
        (disk is None)
        # there are no points in the arm
        or (arm_points is not None and len(arm_points) < 2)
        # the brightness or spread is zero
        or (params['I'] <= 0 or params['spread'] <= 0)
    ):
        return np.zeros(image_size)

    cx, cy = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))

    disk_arr = sersic(
        cx, cy,
        **{**disk, 'I': disk['I'], 'Re': disk['Re'] / params.get('falloff', 1)},
    )
    if distances is None:
        distances = vmap_polyline_distance(
            arm_points,
            cx, cy
        )

    return (
        params['I']
        * np.exp(-distances**2 / (2*params['spread']**2))
        * disk_arr
    )
