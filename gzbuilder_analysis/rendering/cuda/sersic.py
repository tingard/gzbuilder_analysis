import cupy as cp
from scipy.special import gamma
from ..__oversample import oversampled_function


def _b(n):
    # from https://arxiv.org/abs/astro-ph/9911078
    return 2 * n - 1/3 + 4/405/n \
        + 46/25515/n/n + 131/1148175/n**3 \
        - 2194697/30690717750/n**4


def __rotation_matrix(a):
    return cp.array(((cp.cos(a), cp.sin(a)), (-cp.sin(a), cp.cos(a))))


def sersic(x, y, mux=0, muy=0, roll=0, q=1, c=2, I=1, Re=1, n=1):
    # negative of roll as we are looking backwards for the correct radial value
    rm = __rotation_matrix(roll)
    qm = cp.array(((1/q, 0), (0, 1)))
    mu = np.array((mux, muy))
    P = cp.stack((x.ravel(), y.ravel()))
    dp = (cp.expand_dims(mu, 1) - P)
    R = cp.power(
        cp.sum(
            cp.power(cp.abs(cp.dot(qm, cp.dot(rm, dp))), c),
            axis=0
        ),
        1/c
    ).reshape(x.shape)
    intensity = I * cp.exp(-(_b(n) * ((R / Re)**(1/n)) - 1))
    return intensity


def sersic_ltot(I, Re, n):
    return (
        2 * cp.pi * I * Re**2 * n
        * cp.exp(_b(n)) / _b(n)**(2 * n)
        * gamma(2.0 * n)
    )


def sersic_I(L, Re, n):
    return L / (
        2 * cp.pi * Re**2 * n
        * cp.exp(_b(n)) / _b(n)**(2 * n)
        * gamma(2.0 * n)
    )


__oversampled_sersic = oversampled_function(sersic, cp)


def oversampled_sersic_component(comp, image_size=(256, 256), oversample_n=5,
                                 **kwargs):
    if comp is None or comp['I'] == 0:
        return cp.zeros(image_size)
    return __oversampled_sersic(
        shape=image_size, oversample_n=oversample_n, **comp
    )
