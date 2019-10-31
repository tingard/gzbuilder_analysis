import re
import pandas as pd
import numpy as np
from copy import deepcopy
try:
    from cupy import asnumpy
except ModuleNotFoundError:
    asnumpy = np.asarray
from scipy.signal import convolve2d
from gzbuilder_analysis.parsing import to_pandas, from_pandas, sanitize_pandas_params
import gzbuilder_analysis.config as cfg
try:
    import gzbuilder_analysis.rendering.cuda as rg
    from gzbuilder_analysis.rendering.cuda.spiral import spiral_distance
except ModuleNotFoundError:
    import gzbuilder_analysis.rendering as rg
    from gzbuilder_analysis.rendering.spiral import spiral_distance


class Model():
    def __init__(self, model, galaxy_data, psf=None, sigma_image=None,
                 cancel_initial_render=False):
        self.__original_model = deepcopy(model)
        self.data = galaxy_data
        self.psf = psf
        self.sigma_image = sigma_image
        self.nspirals = len(model['spiral'])
        self.__define_params()
        # populate self.params with the provided model
        params = to_pandas(model)
        self.params[params.index] = params
        self.spiral_points = np.array([points for points, params in model['spiral']])
        if len(model['spiral']) > 0:
            self.spiral_distances = [
                spiral_distance(points, distances=np.zeros_like(galaxy_data))
                for points, params in model['spiral']
            ]
        else:
            self.spiral_distances = []
        self._cache = pd.Series(dict(
            disk=0,
            bulge=0,
            bar=0,
            spiral=0,
        ), name='cache', dtype=object)
        # populate the cache
        if not cancel_initial_render:
            self.render(force=True)

    def __define_params(self):
        __model_index = pd.MultiIndex.from_tuples(
            [
                (comp, param) for comp in cfg.ALL_PARAMS.keys()
                for param in cfg.ALL_PARAMS[comp] if comp is not 'spiral'
            ] + [
                (f'spiral{i}', param)
                for i in range(self.nspirals)
                for param in cfg.ALL_PARAMS['spiral']
            ],
            names=('component', 'parameter')
        )
        self.params = pd.Series(np.nan, index=__model_index)

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)

    def __getitem__(self, item):
        return from_pandas(self.params).get(item, None)

    def to_dict(self, params=None):
        if params is None:
            params = self.params
        return from_pandas(
            sanitize_pandas_params(params),
            spirals=self.spiral_points
        )

    def render(self, params=None, model=None, force=False, oversample_n=5):
        if params is not None:
            pass
        elif model is not None:
            params = to_pandas(model)
        else:
            params = self.params
        if force:  # render everything
            components = {'disk', 'bulge', 'bar', 'spiral'}
        else:  # only render components that have changed
            keep_idx = (self.params - params).where(lambda v: v != 0).dropna().index
            components = {
                re.sub(r'[0-9]+', '', i) for i in
                keep_idx.levels[0][
                    keep_idx.codes[0]
                ]
            }
        if 'disk' in components:
            components |= {'spiral'}
        # update the cached params
        self.params[params.index] = params
        for k in components:
            if 'spiral' not in k:
                try:
                    self._cache[k] = rg.oversampled_sersic_component(
                        self.params.dropna()[k].to_dict(),
                        image_size=self.data.shape,
                        oversample_n=oversample_n,
                    )
                except KeyError:
                    self._cache[k] = 0
            else:
                self._cache['spiral'] = sum([
                    rg.spiral_arm(
                        distances=self.spiral_distances[i],
                        params=self.params[f'spiral{i}'].to_dict(),
                        disk=self.params['disk'].to_dict(),
                        image_size=self.data.shape,
                    )
                    for i in range(len(self.spiral_distances))
                ])
        result = np.zeros(self.data.shape) + asnumpy(sum(self._cache.values))
        if self.psf is not None:
            result = convolve2d(result, self.psf, mode='same', boundary='symm')
        return result

    def sanitize(self, force=True):
        _p = self.params.copy
        r = self.render()
        self.params = sanitize_pandas_params(self.params)
        # update the cache
        r2 = self.render()
        if not force and not np.allclose(r, r2):
            print('Model changed during sanitization, aborting')
            self.params = _p
            self.render()
        return self.params

    def reset_params(self):
        params = to_pandas(self.__original_model)
        self.params[params.index] = params
        self.render(force=True)
