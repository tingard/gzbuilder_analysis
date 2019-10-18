import numpy as np
from numba import jit


@jit(nopython=True)
def calc_rolled_coordinates(x, y, mux, muy, roll):
    xp = x * np.cos(roll) - y * np.sin(roll) - mux * np.cos(roll) + muy * np.sin(roll)
    yp = x * np.sin(roll) + y * np.cos(roll) - mux * np.sin(roll) - muy * np.cos(roll)
    return xp, yp


@jit(nopython=True)
def calc_r(xp, yp, mux, muy, q, c):
    return np.power(
        np.power(np.abs(xp - mux), c) + np.power(np.abs((yp - muy) / q), c),
        1 / c
    )


@jit(nopython=True)
def boxy_rolled_radius(x, y, mux, muy, roll, q, c):
    xp, yp = calc_rolled_coordinates(x, y, mux, muy, roll)
    return calc_r(xp, yp, mux, muy, q, c)


@jit(nopython=True)
def _b(n):
    # from https://arxiv.org/abs/astro-ph/9911078
    return 2 * n - 1/3 + 4/405/n \
        + 46/25515/n/n + 131/1148175/n**3 \
        - 2194697/30690717750/n**4


@jit(nopython=True)
def sersic2d(x=0, y=0, mux=0, muy=0, roll=0, Re=1, q=1, c=2, I=1, n=1):
    # https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S132335800000388X
    # note that we rename q from above to q here
    r = boxy_rolled_radius(x, y, mux, muy, roll, q, c)
    return I * np.exp(-_b(n) * np.power(r / Re, 1.0/n))


def oversampled_sersic_component(comp, image_size=(256, 256), oversample_n=1):
    if comp is None:
        return np.zeros(image_size)
    dsx = np.linspace(
        0.5/oversample_n - 0.5,
        image_size[0] - 0.5 - 0.5/oversample_n,
        image_size[0]*oversample_n
    )
    dsy = np.linspace(
        0.5/oversample_n - 0.5,
        image_size[1] - 0.5 - 0.5/oversample_n,
        image_size[1]*oversample_n
    )
    cx, cy = np.meshgrid(dsx, dsy)
    return sersic2d(
        cx, cy, **comp
    ).reshape(
        image_size[0], oversample_n, image_size[1], oversample_n,
    ).mean(3).mean(1)
