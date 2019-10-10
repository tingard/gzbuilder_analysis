import numpy as np
from numba import jit


# image manipulation
@jit(nopython=True)
def asinh(px):
    return np.log(px + np.sqrt(1.0 + (px * px)))


@jit(nopython=True)
def asinh_stretch(px, i=0.6):
    return asinh(px / i) / asinh(i)


# rendering functions
@jit(nopython=True)
def calc_boxy_ellipse_dist(x, y, mux, muy, roll, rEff, axRatio, c):
    xPrime = x * np.cos(roll) \
        - y * np.sin(roll) + mux \
        - mux * np.cos(roll) + muy * np.sin(roll)
    yPrime = x * np.sin(roll) \
        + y * np.cos(roll) + muy \
        - muy * np.cos(roll) - mux * np.sin(roll)
    # return a scaled version of the radius (multiplier is chosen so svg tool
    # doesn't impact badly on shown model component)
    return 3.0 * np.power(
        np.power(axRatio / rEff, c) * np.power(np.abs(xPrime - mux), c)
        + np.power(np.abs(yPrime - muy), c) / np.power(rEff, c),
        1 / c
    )


@jit(nopython=True)
def _b(n):
    # from https://arxiv.org/abs/astro-ph/9911078
    return 2 * n - 1/3 + 4/405/n \
        + 46/25515/n/n + 131/1148175/n**3 \
        - 2194697/30690717750/n**4


@jit(nopython=True)
def sersic2d(x=0, y=0, mux=0, muy=0, roll=0, rEff=1, axRatio=1, c=2, i0=1, n=1):
    # https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S132335800000388X
    return 0.5 * i0 * np.exp(
        _b(n) * (1 - np.power(
            calc_boxy_ellipse_dist(x, y, mux, muy, roll, rEff, axRatio, c),
            1.0 / n
        ))
    )


def sersic_component(comp, x, y):
    return sersic2d(x=x, y=y, **comp)


def oversampled_sersic_component(comp, image_size=512, oversample_n=1):
    if comp is None:
        return np.zeros((image_size, image_size))
    ds = np.linspace(
        0.5/oversample_n - 0.5,
        image_size - 0.5 - 0.5/oversample_n,
        image_size*oversample_n
    )
    cx, cy = np.meshgrid(ds, ds)
    return sersic_component(
        comp, cx, cy,
    ).reshape(
        image_size, oversample_n, image_size, oversample_n,
    ).mean(3).mean(1)
