import jax.numpy as np
from jax import jit
from jax.lax import conv, lgamma
from jax.config import config

config.update("jax_enable_x64", True)


@jit
def _b(n):
    # from https://arxiv.org/abs/astro-ph/9911078
    return 2 * n - 1/3 + 4/405/n \
        + 46/25515/n/n + 131/1148175/n**3 \
        - 2194697/30690717750/n**4


@jit
def roll_coordinates(x, y, mux, muy, roll):
    xp = x * np.cos(roll) + y * np.sin(roll) \
        + mux - mux * np.cos(roll) - muy * np.sin(roll)
    yp = - x * np.sin(roll) + y * np.cos(roll) \
        + muy + mux * np.sin(roll) - muy * np.cos(roll)
    return xp, yp


@jit
def boxy_radius(xp, yp, mux, muy, q, c):
    return np.power(
        np.power(np.abs(xp - mux) / q, c) + np.power(np.abs((yp - muy)), c),
        1 / c
    )


@jit
def sersic(x, y, mux=0, muy=0, roll=0, q=1, c=2, Ie=1, Re=1, n=1):
    xp, yp = roll_coordinates(x, y, mux, muy, roll)
    r = boxy_radius(xp, yp, mux, muy, q, c)
    return Ie * np.exp(-(_b(n) * ((r / Re)**(1/n)) - 1))


@jit
def psf_conv(arr, psf):
    return conv(
        arr.reshape(1, 1, *arr.shape),
        psf.reshape(1, 1, *psf.shape),
        (1, 1),
        'SAME'
    )[0, 0]


@jit
def sersic_ltot(I, Re, n):
    return (
        2 * np.pi * I * Re**2 * n
        * np.exp(_b(n)) / _b(n)**(2 * n)
        * np.exp(lgamma(2.0 * n))
    )


@jit
def sersic_I(L, Re, n):
    return L / (
        2 * np.pi * Re**2 * n
        * np.exp(_b(n)) / _b(n)**(2 * n)
        * np.exp(lgamma(2.0 * n))
    )
