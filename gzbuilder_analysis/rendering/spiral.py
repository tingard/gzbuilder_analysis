import numpy as np
from numba import jit, prange
from gzbuilder_analysis.rendering.sersic import sersic_component
from gzbuilder_analysis.config import DEFAULT_DISK, DEFAULT_SPIRAL


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


def spiral_arm(arm_points, params=DEFAULT_SPIRAL, disk=DEFAULT_DISK,
               image_size=512, arm_distances=None):
    if disk is None or len(arm_points) < 2:
        return np.zeros((image_size, image_size))

    cx, cy = np.meshgrid(np.arange(image_size), np.arange(image_size))

    disk_arr = sersic_component(
        {**disk, 'i0': 1, 'rEff': disk['rEff'] / params['falloff']},
        cx, cy
    )
    if arm_distances is None:
        arm_distances = spiral_distance_numba(
            arm_points,
            distances=np.zeros_like(disk_arr),
        )

    return (
        params['i0']
        * np.exp(-arm_distances**2 * 0.1 / max(params['spread'], 1E-10))
        * disk_arr
    )
