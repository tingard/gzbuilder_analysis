import numpy as np
import cupy as cp
from numba import jit, prange
from .sersic import sersic2d
from gzbuilder_analysis.config import DEFAULT_DISK, DEFAULT_SPIRAL


# TODO: ideally rewrite this to use CUDA
@jit(nopython=True, parallel=True)
def spiral_distance_numba(poly_line, distances=np.zeros((100, 100))):
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


def spiral_arm(arm_points=None, distances=None, params=DEFAULT_SPIRAL, disk=DEFAULT_DISK,
               image_size=(256, 256), return_numpy=False):
    if arm_points is None and distances is None:
        raise TypeError(
            'Must provide either an (N,2) array of points,'
            'or an (N, M) distance matrix'
        )
    if (
        (disk is None)
        or (arm_points is not None and len(arm_points) < 2)
        or (distances is not None and np.all(distances == 0))
        or (params['I'] <= 0 or params['spread'] <= 0)
    ):
        if return_numpy:
            return np.zeros(image_size)
        return cp.zeros(image_size)

    cx, cy = cp.meshgrid(cp.arange(image_size[1]), cp.arange(image_size[0]))
    disk_arr = sersic2d(
        cx, cy,
        **{**disk, 'I': 1, 'Re': disk['Re'] / params['falloff']},
    )
    if distances is None:
        distances = spiral_distance_numba(
            arm_points,
            distances=np.zeros(image_size),
        )
    gpu_distances = cp.asarray(distances)
    render = (
        params['I']
        * cp.exp(-gpu_distances**2 * 0.1 / max(params['spread'], 1E-10))
        * disk_arr
    )
    if return_numpy:
        return cp.asnumpy(render)
    return render
