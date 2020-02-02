import numpy as np
from numba import jit
from scipy.special import gamma
from ..__oversample import oversampled_function


def _b(n):
    # from https://arxiv.org/abs/astro-ph/9911078
    return 2 * n - 1/3 + 4/405/n \
        + 46/25515/n/n + 131/1148175/n**3 \
        - 2194697/30690717750/n**4


def __rotation_matrix(a):
    return np.array(((np.cos(a), np.sin(a)), (-np.sin(a), np.cos(a))))


def sersic(x, y, mux=0.0, muy=0.0, roll=0.0, q=1.0, c=2.0, I=1.0, Re=1.0, n=1.0):
    # negative of roll as we are looking backwards for the correct radial value
    rm = __rotation_matrix(-roll)
    qm = np.array(((q, 0), (0, 1)))
    mu = np.array((muy, mux))
    P = np.stack((x.ravel(), y.ravel()))
    dp = (np.expand_dims(mu, 1) - P)
    R = np.sum(np.dot(qm, np.dot(rm, dp))**c, axis=0)**(1/c)
    intensity = I * np.exp(-(_b(n) * ((R / Re)**(1/n)) - 1))
    return intensity.reshape(x.shape)


def sersic_ltot(I, Re, n):
    return (
        2 * np.pi * I * Re**2 * n
        * np.exp(_b(n)) / _b(n)**(2 * n)
        * gamma(2.0 * n)
    )


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
