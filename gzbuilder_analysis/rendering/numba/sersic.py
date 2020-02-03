import numpy as np
from numba import njit
from scipy.special import gamma
from ..numpy.sersic import _b as __np_b,\
    __rotation_matrix as __rotation_matrix
from ..__oversample import oversampled_function


_b = njit()(__np_b)
__rotation_matrix = njit()(__rotation_matrix)


@njit
def sersic(x, y, mux=0.0, muy=0.0, roll=0.0, q=1.0, c=2.0, I=1.0, Re=1.0, n=1.0):
    # negative of roll as we are looking backwards for the correct radial value
    rm = __rotation_matrix(roll)
    qm = np.array(((1/q, 0), (0, 1)))
    mu = np.array((mux, muy))
    P = np.stack((x.ravel(), y.ravel()))
    dp = (np.expand_dims(mu, 1) - P)
    R = np.power(
        np.sum(
            np.power(np.abs(np.dot(qm, np.dot(rm, dp))), c),
            axis=0
        ),
        1/c
    ).reshape(x.shape)
    intensity = I * np.exp(-(_b(n) * ((R / Re)**(1/n)) - 1))
    return intensity


@njit
def sersic_ltot(I, Re, n):
    return (
        2 * np.pi * I * Re**2 * n
        * np.exp(_b(n)) / _b(n)**(2 * n)
        * gamma(2.0 * n)
    )


@njit
def sersic_I(L, Re, n):
    return L / (
        2 * np.pi * Re**2 * n
        * np.exp(_b(n)) / _b(n)**(2 * n)
        * gamma(2.0 * n)
    )


__oversampled_sersic = oversampled_function(sersic, np)


def oversampled_sersic_component(comp, image_size=(256, 256), oversample_n=5,
                                 **kwargs):
    if comp is None or comp['I'] == 0:
        return np.zeros(image_size)
    return __oversampled_sersic(
        shape=image_size, oversample_n=oversample_n, **comp
    )
