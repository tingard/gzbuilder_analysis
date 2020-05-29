import re
import pandas as pd
import jax.numpy as jnp
from jax import jit
from jax.lax import conv
from .sersic import sersic_I, sersic
from .spiral import translate_spiral, \
    corrected_inclined_lsp, vmap_polyline_distance, correct_logsp_params
from ..parsing.reparametrization import to_reparametrization
from . import _make_xy_arrays, render_comps, psf_conv
from gzbuilder_analysis import df_to_dict
from gzbuilder_analysis.config import COMPONENT_PARAM_BOUNDS
from gzbuilder_analysis.errors import InvalidModelError


class Renderer():
    def __init__(self, psf=None, shape=(100, 100), oversample_n=5):
        """
        model is Pandas series with multiindex
        shape is result of ndarray target.shape
        oversample_n is level of oversampling to be done
        """
        self.psf = psf
        self.shape = shape
        self.oversample_n = oversample_n
        self.P, self.P_super = _make_xy_arrays(shape, oversample_n)

    def check_model(self, model, is_reparametrized=False, include_bounds=False):
        model.index
        if is_reparametrized:
            required_params = {
                'disk':   {'mux', 'muy', 'q', 'roll', 'Re', 'L'},
                'bulge':  {'mux', 'muy', 'q', 'roll', 'scale', 'frac', 'n'},
                'bar':    {'mux', 'muy', 'q', 'roll', 'scale', 'frac', 'n', 'c'},
                'spiral': {'spread', 't_max', 'I', 't_min', 'phi', 'A'},
            }
        else:
            required_params = {
                'disk':   {'mux', 'muy', 'q', 'roll', 'Re', 'I'},
                'bulge':  {'mux', 'muy', 'q', 'roll', 'Re', 'I', 'n'},
                'bar':    {'mux', 'muy', 'q', 'roll', 'Re', 'I', 'n', 'c'},
                'spiral': {'spread', 't_max', 'I', 't_min', 'phi', 'A'},
            }
        try:
            for component in model.index.levels[0]:
                # assert this component is valid
                assert component in COMPONENT_PARAM_BOUNDS
                for param in model[component].keys():
                    stripped_param = re.sub(r'\.[0-9]+', '', param)
                    # ensure this parameter belongs in this component
                    assert stripped_param in COMPONENT_PARAM_BOUNDS[component]
                if component == 'spiral':
                    assert required_params[component].issubset({
                        re.sub(r'\.[0-9]+', '', param)
                        for param in model[component].keys()
                    })
                else:
                    assert required_params[component].issubset(model[component].keys())
        except ValueError:
            return False
        return True

    def __call__(self, model, oversample_n=None, is_reparametrized=False):
        if not isinstance(model, pd.Series):
            model = pd.DataFrame(model).unstack().dropna().astype(jnp.float64)
        if not self.check_model(model):
            raise InvalidModelError('Model specification was not valid')
        # convert to reparametrization
        if not is_reparametrized:
            model = to_reparametrization(model, self.shape)
        # # render components
        has_bulge = ('bulge', 'n') in model
        has_bar = ('bar', 'c') in model
        try:
            n_arms = sum(1 for i in model['spiral'].index if 'I.' in i)
        except ValueError:
            n_arms = 0
        rendered_components = render_comps(
            model.to_dict(), has_bulge, has_bar, n_arms, self.shape,
            self.oversample_n, model.get(('disk', 'roll'), 0)
        )
        # # PSF convolve
        if self.psf is not None:
            return psf_conv(sum(rendered_components.values()), self.psf)
        return sum(rendered_components.values())
