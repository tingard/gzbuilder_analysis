import jax.numpy as np
from jax import vmap
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


def spiral_from_polyline(x, y, disk, points, params):
    distances = vmap_polyline_distance(points, x, y)
    return (
        params['I']
        * np.exp(-distances**2 * 0.1 / params['spread'])
        * disk
    )


def spiral_from_distances(disk, distances, params):
    return (
        params['I']
        * np.exp(-distances**2 * 0.1 / params['spread'])
        * disk
    )
