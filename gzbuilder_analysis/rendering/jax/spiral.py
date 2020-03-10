import jax.numpy as np
from jax import vmap
from jax import jit
from jax.config import config
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


# def spiral_from_polyline(x, y, disk, points, params):
#     distances = vmap_polyline_distance(points, x, y)
#     return (
#         params['I']
#         * np.exp(-distances**2 / (2*params['spread']**2))
#         * disk
#     )
#
#
# def spiral_from_distances(disk, distances, params):
#     return (
#         params['I']
#         * np.exp(-distances**2 / (2*params['spread']**2))
#         * disk
#     )
