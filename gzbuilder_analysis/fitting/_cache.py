import numpy as np
from gzbuilder_analysis.fitting._model import Model


class CachedModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_template = self._template
        self._cached_p = self.to_p(self._model, self._template)
        self.cache = self._calculate_components(self._model)
        self.param_index = {'disk': 0, 'bulge': 1, 'bar': 2, 'spiral': 3}

    def cached_render(self, new_model, template=None):
        template = self.sanitize_template(template)
        idx = np.arange(len(template))
        new_p = self.to_p(new_model)
        diff = new_p != self._cached_p
        params_to_update = np.unique([template[i][0] for i in idx[diff]])

        for param in params_to_update:
            self.cache[self.param_index[param]] = (
                self._render_component(param, new_model)
            )
        self._cached_template = template
        self._cached_p = new_p

        return self._psf_convolve(np.add.reduce(list(self.cache), axis=0))
