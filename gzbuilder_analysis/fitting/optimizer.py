from copy import deepcopy
import pandas as pd
import jax.numpy as np
from jax import jit, jacrev
from ..rendering.jax.sersic import sersic_I, sersic
from ..rendering.jax.spiral import translate_spiral, \
    corrected_inclined_lsp, vmap_polyline_distance, correct_logsp_params
from .misc import _make_xy_arrays, downsample, psf_conv
from .nnlf import negative_log_likelihood
from .reparametrization import to_reparametrization, \
    get_reparametrized_errors, get_limits


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
            c=2.0
        )
        out['bar'] = np.squeeze(downsample(bar_super, oversample_n))

    # return the dictionary of rendered components
    return out


render_comps = jit(__render_comps, static_argnums=(1, 2, 3, 4, 5, 6))


def __get_nnlf(
    model, has_bulge, has_bar, n_spirals, shape, oversample_n, base_roll,
    base_model, model_err, psf, mask, target, sigma_image
):
    """Calculate the Galaxy Builder optimization objective function: where we
    assume gaussian error on pixel values, and impose a prior over model
    parameters obtained through clustering (position, size, ellipticity and
    rotation)

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
    base_model -- a dictionary of the same type as model, containing the means
        of the Normal priors on parameters
    model_err --  a dictionary of the same type as model, containing the sigmas
        of the Normal priors on parameters
    psf -- the PSF of the galaxy, measured using SDSS
    mask -- the deblending mask obtained using SExtractor
    target -- the stacked SDSS r-band image we are trying to recreate
    sigma_image -- the pixel uncertainty for the stacked image
    """
    # render the model's components
    rendered_components = render_comps(
        model, has_bulge, has_bar, n_spirals, shape, oversample_n, base_roll
    )
    # sum these components and convolve with the PSF
    rendered_image = psf_conv(sum(rendered_components.values()), psf)

    # Mask, subtract the target and scale by the sigma image
    masked_normalised_render_err = (
        rendered_image[~mask] - target[~mask]
    ) / sigma_image[~mask]

    # centre and scale the parameter errors, where the error is not inf
    scaled_param_err = np.asarray([
        (model[k] - base_model[k]) / model_err[k]
        for k in model.keys()
        if np.isfinite(model_err[k])
    ])

    # return the summed negative log likelihood for the data and the parameters
    return negative_log_likelihood(
        np.concatenate((masked_normalised_render_err.ravel(), scaled_param_err))
    )


get_nnlf = jit(
    __get_nnlf,
    static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
)


class Optimizer():
    def __init__(self, aggregation_result, psf, galaxy_data, sigma_image,
                 oversample_n=5):
        self.aggregation_result = aggregation_result
        # reparametrize the model
        self.model = pd.DataFrame(
            to_reparametrization(aggregation_result.model)
        ).unstack().dropna()
        # save this as a dictionary (for fitting)
        self.model_ = self.model.to_dict()

        # we need to also save the initial parameters
        self.base_model = self.model.copy()
        self.base_model_ = self.base_model.to_dict()

        # save the roll of the original disk component
        self.base_roll = aggregation_result.model['disk']['roll']

        # reparametrize parameter error
        self.model_err = pd.DataFrame(
            get_reparametrized_errors(aggregation_result)
        ).unstack().dropna()

        # save a dictionary version of this error
        self.model_err_ = self.model_err.to_dict()

        # get the parameter limits to use
        self.lims = get_limits(aggregation_result)
        self.lims_ = pd.DataFrame(self.lims).unstack().dropna()

        # save the list of keys present in the model (useful for preserving
        # ordering)
        self.keys = list(self.model.keys())

        # save the rest of the metadata needed for fitting
        self.psf = psf
        self.oversample_n = oversample_n
        self.n_spirals = len(self.aggregation_result.spiral_arms)
        self.target = np.asarray(galaxy_data.data)
        self.mask = np.asarray(galaxy_data.mask)
        self.sigma = np.asarray(sigma_image.data)

        # compile the objective and jacobian functions for speedy computation
        self.__call__ = jit(self.__call__, static_argnums=(1,))
        self.jac = jit(jacrev(self.__call__), static_argnums=(1,))

    def __call__(self, p, keys):
        new_model = {
            **deepcopy(self.model_),
            **{k: v for k, v in zip(keys, p)}
        }
        # Important not to get the argument ordering wrong...
        return get_nnlf(
            new_model,
            ('bulge', 'n') in new_model,
            ('bar', 'c') in new_model,
            self.n_spirals,
            self.target.shape,
            self.oversample_n,
            self.base_roll,
            self.base_model_,
            self.model_err_,
            self.psf,
            self.mask,
            self.target,
            self.sigma
        )

    def render_comps(self, model=None, correct_spirals=True):
        model = model if model is not None else self.model_
        if type(model) != dict:
            model = model.to_dict()
        n_spirals = sum(1 for i in model if i[0] == 'spiral') // 6
        return render_comps(
            model,
            ('bulge', 'n') in model,
            ('bar', 'c') in model,
            n_spirals,
            self.target.shape,
            self.oversample_n,
            (self.base_roll if correct_spirals else model[('disk', 'roll')])
        )

    def update(self, other):
        self.model = self.model.update(other)
        self.model_ = self.model.to_dict()

    def __setitem__(self, key, val):
        self.model[key] = val
        self.model_ = self.model.to_dict()

    def reset(self):
        return Optimizer(
            self.aggregation_result,
            self.psf,
            self.target,
            self.sigma,
            oversample_n=self.oversample_n
        )
