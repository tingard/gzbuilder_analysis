import numpy as np
import cupy as cp
from numba import jit, prange
from .sersic import sersic2d
from gzbuilder_analysis.rendering.spiral import spiral_distance as __spiral_distance_numba
from gzbuilder_analysis.config import DEFAULT_DISK, DEFAULT_SPIRAL


# TODO: ideally rewrite this to use GPU not CPU (custom kernel?)
def spiral_distance(*args, **kwargs):
    return cp.asarray(__spiral_distance_numba(*args, **kwargs))


def spiral_arm(arm_points=None, distances=None, params=DEFAULT_SPIRAL, disk=DEFAULT_DISK,
               image_size=(256, 256)):
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
        return 0

    cx, cy = cp.meshgrid(cp.arange(image_size[1]), cp.arange(image_size[0]))
    disk_arr = sersic2d(
        cx, cy,
        **{**disk, 'I': 1, 'Re': disk['Re'] / params['falloff']},
    )
    if distances is None:
        distances = spiral_distance(
            arm_points,
            distances=np.zeros(image_size),
        )
    if type(distances) == np.ndarray:
        distances = cp.asarray(distances)
    render = (
        params['I']
        * cp.exp(-distances**2 / (2*params['spread']**2))
        * disk_arr
    )
    return render
