import numpy as np
from numba import jit, prange


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
