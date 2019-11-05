import numpy as np
from numba import jit
from .__oversample import oversampled_function


@jit(nopython=True)
def roll_coordinates(x, y, mux, muy, roll):
    xp = x * np.cos(roll) + y * np.sin(roll) + mux - mux * np.cos(roll) - muy * np.sin(roll)
    yp = - x * np.sin(roll) + y * np.cos(roll) + muy + mux * np.sin(roll) - muy * np.cos(roll)
    return xp, yp


@jit(nopython=True)
def boxy_radius(xp, yp, mux, muy, q, c):
    return np.power(
        np.power(np.abs(xp - mux) / q, c) + np.power(np.abs((yp - muy)), c),
        1 / c
    )


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
    r = boxy_radius(
        *roll_coordinates(x, y, mux, muy, roll),
        mux, muy, q, c,
    )
    return I * np.exp(-_b(n) * (np.power(r / Re, 1.0/n) - 1))


__oversampled_sersic = oversampled_function(sersic2d)


def oversampled_sersic_component(comp, image_size=(256, 256), oversample_n=5, **kwargs):
    if comp is None:
        return np.zeros(image_size)
    return __oversampled_sersic(
        shape=image_size, oversample_n=oversample_n, **comp
    )
