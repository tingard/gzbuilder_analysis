import numpy as np
import cupy as cp


def calc_rolled_coordinates(x, y, mux, muy, roll):
    _p = cp.get_array_module(x)
    xp = x * _p.cos(roll) + y * _p.sin(roll) + mux - mux * _p.cos(roll) - muy * _p.sin(roll)
    yp = - x * _p.sin(roll) + y * _p.cos(roll) + muy + mux * _p.sin(roll) - muy * _p.cos(roll)
    return xp, yp


def calc_r(xp, yp, mux, muy, q, c):
    _p = cp.get_array_module(xp)
    return _p.power(
        _p.power(_p.abs(xp - mux) / q, c) + _p.power(_p.abs((yp - muy)), c),
        1 / c
    )

def boxy_rolled_radius(x, y, mux, muy, roll, q, c):
    xp, yp = calc_rolled_coordinates(x, y, mux, muy, roll)
    return calc_r(xp, yp, mux, muy, q, c)


def _b(n):
    # from https://arxiv.org/abs/astro-ph/9911078
    return 2 * n - 1/3 + 4/405/n \
        + 46/25515/n/n + 131/1148175/n**3 \
        - 2194697/30690717750/n**4


def sersic2d(x=0, y=0, mux=0, muy=0, roll=0, Re=1, q=1, c=2, I=1, n=1):
    # https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S132335800000388X
    # note that we rename q from above to q here
    r = boxy_rolled_radius(x, y, mux, muy, roll, q, c)
    _p = cp.get_array_module(r)
    return I * _p.exp(-_b(n) * (_p.power(r / Re, 1.0/n) - 1))


def oversampled_sersic_component(comp, image_size=(256, 256), oversample_n=5, cuda=True):
    if cuda:
        _p = cp
    else:
        _p = np
    if comp is None:
        return cp.zeros(image_size)
    dsx = _p.linspace(
        0.5/oversample_n - 0.5,
        image_size[0] - 0.5 - 0.5/oversample_n,
        image_size[0]*oversample_n
    )
    dsy = _p.linspace(
        0.5/oversample_n - 0.5,
        image_size[1] - 0.5 - 0.5/oversample_n,
        image_size[1]*oversample_n
    )
    cx, cy = _p.meshgrid(dsx, dsy)
    return sersic2d(
        cx, cy, **comp
    ).reshape(
        image_size[0], oversample_n, image_size[1], oversample_n,
    ).mean(3).mean(1)