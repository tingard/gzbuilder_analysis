import jax.numpy as np
from jax import jit
from jax.lax import conv
from .sersic import sersic_I, sersic
from .spiral import translate_spiral, \
    corrected_inclined_lsp, vmap_polyline_distance, correct_logsp_params


def _make_xy_arrays(shape, On):
    x = np.arange(shape[1], dtype=np.float64)
    y = np.arange(shape[0], dtype=np.float64)
    cx, cy = np.meshgrid(x, y)
    x_super = np.linspace(0.5 / On - 0.5, shape[1] - 0.5 - 0.5 / On,
                          shape[1] * On)
    y_super = np.linspace(0.5 / On - 0.5, shape[0] - 0.5 - 0.5 / On,
                          shape[0] * On)
    cx_super, cy_super = np.meshgrid(x_super, y_super)
    return (cx, cy), (cx_super, cy_super)


def downsample(arr, n=5):
    """downsample an array of (n*x, m*y, m) into (x, y, m) using the mean
    """
    shape = (np.asarray(arr.shape) / n).astype(int)
    return arr.reshape(shape[0], n, shape[1], n, -1).mean(3).mean(1)


downsample = jit(downsample, static_argnums=(1,))


@jit
def psf_conv(arr, psf):
    return conv(
        arr.reshape(1, 1, *arr.shape),
        psf.reshape(1, 1, *psf.shape),
        (1, 1),
        'SAME'
    )[0, 0]


def __get_spirals(model, n_spirals, base_roll):
    dpsi = model[('disk', 'roll')] - base_roll
    return [
        translate_spiral(
            corrected_inclined_lsp(
                # (A, phi, q, psi, dpsi, theta)
                model[('spiral', 'A.{}'.format(i))],
                model[('spiral', 'phi.{}'.format(i))],
                model[('disk', 'q')],
                model[('disk', 'roll')],
                dpsi,
                np.linspace(
                    model[('spiral', 't_min.{}'.format(i))],
                    model[('spiral', 't_max.{}'.format(i))],
                    100
                ),
            ),
            model[('disk', 'mux')],
            model[('disk', 'muy')],
        )
        for i in range(n_spirals)
    ]


get_spirals = jit(__get_spirals, static_argnums=(1,))


def __render_comps(model, has_bulge, has_bar, n_spirals, shape,
                   oversample_n, base_roll):
    """Render the components of a galaxy builder model

    Arguments:
    model -- The model to render. This should be a dictionary like
        {(param, component): value} (i.e. what you would get from
        pd.Series(...).to_dict())
    has_bulge -- Whether the model contains a bulge, needed for Jax compilation
    has_bar -- Whether the model contains a bar
    n_spirals -- The number of spiral arms present in the model
    shape -- The desired output shape to render
    oversample_n -- The factor to which Sersic oversampling will be done
    base_roll -- The roll parameter of the original model. This is needed to
        preserve the location of spiral arms as best as possible
    """
    P, P_super = _make_xy_arrays(shape, oversample_n)

    out = {}

    disk_I = sersic_I(
        model[('disk', 'L')], model[('disk', 'Re')],
        model[('disk', 'q')], 1, 2
    )
    disk_super = sersic(
        *P_super,
        mux=model[('disk', 'mux')],
        muy=model[('disk', 'muy')],
        roll=model[('disk', 'roll')],
        q=model[('disk', 'q')],
        Re=model[('disk', 'Re')],
        I=disk_I,
        n=1.0,
        c=2.0,
    )

    out['disk'] = np.squeeze(downsample(disk_super, oversample_n))

    # next add spirals to the disk
    if n_spirals > 0:
        spirals = get_spirals(model, n_spirals, base_roll)
        spiral_distances = np.stack([
            vmap_polyline_distance(s, *P)
            for s in spirals
        ], axis=-1)

        Is = np.array([
            model[('spiral', 'I.{}'.format(i))]
            for i in range(n_spirals)
        ])
        spreads = np.array([
            model[('spiral', 'spread.{}'.format(i))] for i in range(n_spirals)
        ])
        spirals = np.sum(
            Is
            * np.exp(-spiral_distances**2 / (2*spreads**2))
            * np.expand_dims(out['disk'], -1),
            axis=-1
        )
        out['spiral'] = spirals
    else:
        spirals = np.zeros(shape)

    # calculate the luminosity of the disk and spirals together (the bulge and
    # bar fractions are calculated relative to this)
    disk_spiral_L = model[('disk', 'L')] + spirals.sum()

    # if we have a bulge, render it
    if has_bulge:
        bulge_L = (
            model[('bulge', 'frac')] * (disk_spiral_L)
            / (1 - model[('bulge', 'frac')])
        )
        bulge_Re = model[('disk', 'Re')] * model[('bulge', 'scale')]
        bulge_I = sersic_I(
            bulge_L, bulge_Re, model[('bulge', 'q')], model[('bulge', 'n')]
        )
        bulge_super = sersic(
            *P_super,
            mux=model[('centre', 'mux')],
            muy=model[('centre', 'muy')],
            roll=model[('bulge', 'roll')],
            q=model[('bulge', 'q')],
            Re=bulge_Re,
            I=bulge_I,
            n=model[('bulge', 'n')],
            c=2.0
        )
        out['bulge'] = np.squeeze(downsample(bulge_super, oversample_n))

    # if we have a bar, render it
    if has_bar:
        bar_L = (
            model[('bar', 'frac')] * (disk_spiral_L)
            / (1 - model[('bar', 'frac')])
        )
        bar_Re = model[('disk', 'Re')] * model[('bar', 'scale')]
        bar_I = sersic_I(
            bar_L, bar_Re, model[('bar', 'q')], model[('bar', 'n')]
        )
        bar_super = sersic(
            *P_super,
            mux=model[('centre', 'mux')],
            muy=model[('centre', 'muy')],
            roll=model[('bar', 'roll')],
            q=model[('bar', 'q')],
            Re=bar_Re,
            I=bar_I,
            n=model[('bar', 'n')],
            c=model[('bar', 'c')],
        )
        out['bar'] = np.squeeze(downsample(bar_super, oversample_n))

    # return the dictionary of rendered components
    return out


render_comps = jit(__render_comps, static_argnums=(1, 2, 3, 4, 5, 6))


class Renderer():
    def __init__(self, psf, shape, oversample_n):
        """
        model is Pandas series with multiindex
        shape is result of ndarray target.shape
        oversample_n is level of oversampling to be done
        """
        self.psf = psf
        self.shape = shape
        self.oversample_n = oversample_n
        self.P, self.P_super = _make_xy_arrays(shape, oversample_n)

    def __call__(model):
        # convert to reparametrization
        # model = to_reparametrization(model)
        # # render components
        # rendered_components = render_comps(
        #     model, has_bulge, has_bar, n_spirals, shape,
        #     oversample_n, base_roll
        # )
        # # PSF convolve
        # blurred_render = psf_conv(sum(rendered_components.values()), psf)
        # return blurred_render
        pass
