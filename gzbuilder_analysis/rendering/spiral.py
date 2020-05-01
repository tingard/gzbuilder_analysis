import numpy as np
from numba import jit, prange
from .sersic import sersic
from gzbuilder_analysis.config import DEFAULT_DISK, DEFAULT_SPIRAL


@jit(nopython=True, parallel=True)
def spiral_distance(poly_line, distances=np.zeros((100, 100))):
    for i in prange(distances.shape[0]):
        for j in range(distances.shape[1]):
            best = 1E30
            # for each possible pair of vertices
            for k in range(len(poly_line) - 1):
                ux = j - poly_line[k, 0]
                uy = i - poly_line[k, 1]
                vx = poly_line[k + 1, 0] - poly_line[k, 0]
                vy = poly_line[k + 1, 1] - poly_line[k, 1]
                dot = ux * vx + uy * vy
                t = dot / (vx**2 + vy**2)
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                d = (vx*t - ux)**2 + (vy*t - uy)**2
                if d < best:
                    best = d
            distances[i, j] = best
    return np.sqrt(distances)


@jit(nopython=True, parallel=True)
def spiral_distance2(poly_line, x, y):
    min_t = np.zeros_like(x, dtype=np.float64)
    max_t = np.ones_like(x, dtype=np.float64)
    best = np.zeros_like(x, dtype=np.float64) + np.inf
    # for each possible pair of vertices
    for k in range(len(poly_line) - 1):
        ux = y - poly_line[k, 0]
        uy = x - poly_line[k, 1]
        vx = poly_line[k + 1, 0] - poly_line[k, 0]
        vy = poly_line[k + 1, 1] - poly_line[k, 1]
        dot = ux * vx + uy * vy
        t = dot / (vx**2 + vy**2)
        # t = np_clip(t, 0, 1)
        t = np.minimum(max_t, np.maximum(min_t, t))
        d = (vx*t - ux)**2 + (vy*t - uy)**2
        best = np.minimum(best, d)
    return np.sqrt(best)


def spiral_arm(arm_points=None, distances=None, params=DEFAULT_SPIRAL,
               disk=DEFAULT_DISK, image_size=(256, 256)):
    if arm_points is None and distances is None:
        raise TypeError(
            'Must provide either an (N,2) array of points,'
            'or an (N, M) distance matrix'
        )
    if disk is None:
        return np.zeros(image_size)
    if arm_points is not None and len(arm_points) < 2:
        return np.zeros(image_size)
    if distances is not None and np.all(distances == 0):
        return np.zeros(image_size)
    if params['I'] <= 0 or params['spread'] <= 0:
        return np.zeros(image_size)
    cx, cy = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))

    disk_arr = sersic(
        cx, cy,
        **{**disk, 'I': disk['I'], 'Re': disk['Re'] / params['falloff']},
    )
    if distances is None:
        distances = spiral_distance(
            arm_points,
            distances=np.zeros_like(disk_arr),
        )

    return (
        params['I']
        * np.exp(-distances**2 / (2*params['spread']**2))
        * disk_arr
    )
