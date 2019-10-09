import numpy as np
from gzbuilder_analysis.fitting._model import Model


class CachedModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_template = self._template
        self._cached_p = self.to_p(self._model, self._template)
        self.cache = self._calculate_components(self._model)
        self.cache_index = {'disk': 0, 'bulge': 1, 'bar': 2, 'spiral': 3}

    def cached_render(self, new_model=None, template=None):
        new_model = self._model if new_model is None else new_model
        template = self.sanitize_template(template)
        idx = np.arange(len(template))
        new_p = self.to_p(new_model)
        diff = new_p != self._cached_p
        comps_to_update = np.unique([template[i][0] for i in idx[diff]]).tolist()
        # as the spirals depend on the disk, we link them
        if ('disk' in comps_to_update) and ('spiral' not in comps_to_update):
            comps_to_update.append('spiral')
        for comp in comps_to_update:
            self.cache[self.cache_index[comp]] = (
                self._render_component(comp, new_model)
            )
        self._cached_template = template
        self._cached_p = new_p

        return self._psf_convolve(np.add.reduce(list(self.cache), axis=0))
