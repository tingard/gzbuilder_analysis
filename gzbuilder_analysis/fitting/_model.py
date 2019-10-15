import pandas as pd
import numpy as np
from copy import deepcopy
from gzbuilder_analysis.parsing import sanitize_model
import gzbuilder_analysis.rendering as rg
from gzbuilder_analysis.rendering.sersic import oversampled_sersic_component
from gzbuilder_analysis.rendering.spiral import spiral_arm
from gzbuilder_analysis.config import FIT_PARAMS

# template looks like
# [('disk', 'axRatio'), ('disk', 'rEff'), ..., ('spiral', 0, 'i0'), ...]


class Model():
    def __init__(self, model, galaxy_data, psf=None, pixel_mask=None, sigma_image=None):
        self._model = deepcopy(model)
        self._n_spirals = len(self._model['spiral'])
        # TODO: save spiral arm distances to avoid having to recalculate
        # self.distances = ...
        self.data = galaxy_data
        self.psf = psf if psf is not None else np.ones((1, 1))
        self.pixel_mask = (
            pixel_mask if pixel_mask is not None
            else np.ones_like(galaxy_data)
        )
        self.sigma_image = sigma_image
        self._template = [
            (k, v) for k in FIT_PARAMS.keys() for v in FIT_PARAMS[k]
            if k != 'spiral' and self._model[k] is not None
        ] + [
            ('spiral', i, v)
            for i in range(self._n_spirals)
            for v in FIT_PARAMS['spiral']
        ]

    def __getitem__(self, key):
        return self._model.get(key, None)

    def __setitem__(self, key, value):
        self._model[key] = value

    def sanitize_template(self, template):
        if template is None:
            return self._template
        tpl_sersic = [
            v for v in template
            if v[0] != 'spiral' and self[v[0]] is not None
        ]
        spiral_indices_present = any(
            len(v) == 3 for v in template if v[0] == 'spiral'
        )
        if spiral_indices_present:
            tpl_spiral = [v for v in template if v[0] == 'spiral']
        else:
            tpl_spiral = [
                (v[0], i, v[1])
                for i in range(self._n_spirals)
                for v in template
                if v[0] == 'spiral'
            ]
        return tpl_sersic + tpl_spiral

    def to_p(self, model=None, template=None):
        template = self.sanitize_template(template)
        model = model if model is not None else self._model
        try:
            return np.fromiter((
                model[k[0]][k[1]]
                if (k[0] != 'spiral' and model[k[0]])
                else model[k[0]][k[1]][1][k[2]]
                for k in template
            ), count=len(template), dtype=np.float64)
        except TypeError:
            raise TypeError('Invalid template for this model')

    def from_p(self, p, template=None):
        template = self.sanitize_template(template)
        new_model = deepcopy(self._model)
        for i, k in enumerate(template):
            try:
                if k[0] != 'spiral':
                    new_model[k[0]][k[1]] = p[i]
                else:
                    new_model[k[0]][k[1]][1][k[2]] = p[i]
            except (KeyError, TypeError):
                pass
        return sanitize_model(new_model)

    def _render_component(self, comp_name, model, oversample_n=5):
        if comp_name == 'spiral':
            if len(model['spiral']) == 0:
                return np.zeros_like(self.data)
            return np.add.reduce([
                spiral_arm(
                    *s,
                    model['disk'],
                    image_size=self.data.shape,
                )
                for s in model['spiral']
            ])
        if model[comp_name] is None:
            return np.zeros_like(self.data)
        return oversampled_sersic_component(
            model[comp_name],
            image_size=self.data.shape,
            oversample_n=oversample_n,
        )

    def _calculate_components(self, model, oversample_n=5):
        disk_arr = self._render_component('disk', model, oversample_n)
        bulge_arr = self._render_component('bulge', model, oversample_n)
        bar_arr = self._render_component('bar', model, oversample_n)
        spiral_arr = self._render_component('spiral', model, oversample_n)
        return np.stack((disk_arr, bulge_arr, bar_arr, spiral_arr))

    def _psf_convolve(self, arr):
        return rg.convolve2d(
            arr, self.psf, mode='same', boundary='symm'
        )

    def render(self, model=None):
        _model = model if model is not None else self._model
        return self._psf_convolve(
            np.add.reduce(
                self._calculate_components(_model)
            )
        )

    def to_df(self, model=None):
        m = model if model is not None else self._model
        idx = ('disk', 'bulge', 'bar')

        c = pd.DataFrame([m[i] if m[i] is not None else {} for i in idx],
                         index=idx)
        spirals = pd.DataFrame([i[1] for i in m['spiral']])
        spirals.index.name = 'Spiral number'
        return c, spirals

    def _repr_html_(self, model=None):
        c, spirals = self.to_df(model)
        return c.to_html() + spirals.to_html()

    def get_n_params(self, model=None):
        m = self._model if model is None else model
        n = [
            6 + i
            for i, k in enumerate(('disk', 'bulge', 'bar'))
            if m.get(k, False)
        ] + [
            len(s[0])
        ]


    def copy_with_new_model(self, new_model=None):
        return self.__class__(
            new_model, self.data,
            psf=self.psf, pixel_mask=self.pixel_mask,
            sigma_image=self.sigma_image,
        )
