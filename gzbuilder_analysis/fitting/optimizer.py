from copy import deepcopy
import pandas as pd
import jax.numpy as jnp
from jax import jit, jacrev
from .nnlf import negative_log_likelihood
from ..parsing.reparametrization import to_reparametrization, \
    get_reparametrized_errors, get_limits
from ..rendering.renderer import psf_conv, render_comps


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
    scaled_param_err = jnp.asarray([
        (model[k] - base_model[k]) / model_err[k]
        for k in model.keys()
        if jnp.isfinite(model_err[k])
    ])

    # return the summed negative log likelihood for the data and the parameters
    return negative_log_likelihood(
        jnp.concatenate((masked_normalised_render_err.ravel(), scaled_param_err))
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
        if isinstance(aggregation_result.model, dict):
            # using an old version of AggregationResult
            self.model = to_reparametrization(aggregation_result.params, galaxy_data.shape)
        else:
            self.model = to_reparametrization(aggregation_result.model, galaxy_data.shape)

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
        self.target = jnp.asarray(galaxy_data.data)
        self.mask = jnp.asarray(galaxy_data.mask)
        self.sigma = jnp.asarray(sigma_image.data)

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

    def update(self, *args, **kwargs):
        self.model.update(*args, **kwargs)
        self.model_ = self.model.to_dict()

    def drop(self, *args, **kwargs):
        self.model = self.model.drop(*args, **kwargs)
        self.model_ = self.model.to_dict()

    def __getitem__(self, key):
        return self.model[key]

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
