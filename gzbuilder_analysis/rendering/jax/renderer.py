import jax.numpy as np


def make_xy_arrays(shape, oversample_n):
    x = np.arange(shape[1], dtype=np.float64)
    y = np.arange(shape[0], dtype=np.float64)
    x_super = np.linspace(
        0.5 / oversample_n - 0.5,
        shape[1] - 0.5 - 0.5 / oversample_n,
        shape[1] * oversample_n
    )
    y_super = np.linspace(
        0.5 / oversample_n - 0.5,
        shape[0] - 0.5 - 0.5 / oversample_n,
        shape[0] * oversample_n
    )
    return np.meshgrid(x, y), np.meshgrid(x_super, y_super)


def _get_distances(cx, cy, model, n_spirals):
    delta_roll = model['disk']['roll'] - base_model['disk']['roll']
    if n_spirals > 0:
        spirals = [
            # t_min, t_max, A, phi, q, psi, dpsi, mux, muy, N
            logsp(
                model['spiral']['t_min.{}'.format(i)],
                model['spiral']['t_max.{}'.format(i)],
                model['spiral']['A.{}'.format(i)],
                model['spiral']['phi.{}'.format(i)],
                model['disk']['q'],
                model['disk']['roll'],
                delta_roll,
                model['disk']['mux'],
                model['disk']['muy'],
                200,
            )
            for i in range(n_spirals)
        ]
        distances = np.stack([
            vmap_polyline_distance(s, cx, cy)
            for s in spirals
        ], axis=-1)
    else:
        distances = np.array([], dtype=np.float64)
    return distances


class Renderer():
    def __init__(self, model, shape, oversample_n):
        """
        model is Pandas series with multiindex
        shape is result of ndarray target.shape
        oversample_n is level of oversampling to be done
        """
        self.model = model
        self.shape = shape
        self.oversample_n = oversample_n
        self.P, self.P_super = make_xy_arrays(shape, oversample_n)

    def __call__(p, keys):
