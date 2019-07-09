from copy import deepcopy
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from gzbuilder_analysis.config import PARAM_BOUNDS


def get_param_bounds(new_bounds={}):
    return {**PARAM_BOUNDS, **new_bounds}


class ModelFitter():
    def __init__(self, base_model, galaxy_data, psf=None, pixel_mask=None):
        self.model = Model(base_model, galaxy_data, psf, pixel_mask)

    def loss(self, rendered_model):
        pixel_mask = self.model.pixel_mask
        Y = rendered_model * pixel_mask
        return mean_squared_error(
            Y.flatten(),
            0.8 * (self.model.data * pixel_mask).flatten()
        )

    def fit(self, oversample_n=5, bounds={}, *args, **kwargs):
        md = deepcopy(self.model.base_model)
        p0, p_key, bounds = self.model.construct_p(
            md, bounds=get_param_bounds(bounds)
        )

        def _f(p):
            rendered_model = self.model.render_from_p(p)
            if rendered_model.max() > 1E3:
                print(p)
            return self.loss(rendered_model)

        # return _f, p0
        res = minimize(_f, p0, bounds=bounds, *args, **kwargs)
        return self.model.update_model(
            md, res['x']
        ), res
