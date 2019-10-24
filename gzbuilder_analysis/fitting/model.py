import re
import pandas as pd
import numpy as np
try:
    from cupy import asnumpy
except ModuleNotFoundError:
    asnumpy = np.asarray
from copy import deepcopy
from scipy.optimize import minimize
from scipy.signal import convolve2d
from tqdm import tqdm
from gzbuilder_analysis.parsing import sanitize_model
try:
    import gzbuilder_analysis.rendering.cuda as rg
    from gzbuilder_analysis.rendering.cuda.spiral import spiral_distance
except ModuleNotFoundError:
    import gzbuilder_analysis.rendering as rg
    from gzbuilder_analysis.rendering.spiral import spiral_distance
from . import chisq
import gzbuilder_analysis.config as cfg


class Model():
    def __init__(self, model, galaxy_data, psf=None, sigma_image=None,
                 cancel_initial_render=False):
        self.data = galaxy_data
        self.psf = psf
        self.sigma_image = sigma_image
        self.nspirals = len(model['spiral'])
        self.__define_params()
        # populate self.params with the provided model
        params = self.get_params(model)
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

    @staticmethod
    def get_params(model):
        # filter out empty components
        model = {k: v for k, v in model.items() if v is not None}

        params = {
            f'{comp} {param}': model[comp][param]
            for comp in ('disk', 'bulge', 'bar')
            for param in model.get(comp, {}).keys()
        }
        params.update({
            f'spiral{i} {param}': model['spiral'][i][1][param]
            for i in range(len(model['spiral']))
            for param in model['spiral'][i][1].keys()
        })
        idx = pd.MultiIndex.from_tuples([
            k.split() for k in params.keys()
        ], names=('component', 'parameter'))
        vals = [params.get(' '.join(p), np.nan) for p in idx.values]
        return pd.Series(vals, index=idx, name='value')

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)

    def __getitem__(self, item):
        try:
            return self.params[item]
        except KeyError:
            return None

    def to_dict(self, params=None):
        if params is None:
            params = self.params
        spirals = list(zip(
            self.spiral_points,
            [params[f'spiral{i}'].to_dict() for i in range(self.nspirals)]
        ))
        d = dict(
            disk=params['disk'].dropna().to_dict(),
            bulge=params['bulge'].dropna().to_dict(),
            bar=params['bar'].dropna().to_dict(),
            spiral=spirals
        )
        return {k: (v if v != {} else None) for k, v in d.items()}

    def render(self, model=None, params=None, force=False, oversample_n=5):
        if model is None and params is None:
            # render the current model
            params = self.params
        elif params is None:
            # get params from the model provided
            params = self.get_params(model)
        if force:
            # render everything
            components = {'disk', 'bulge', 'bar', 'spiral'}
        else:
            # only render components that have changed
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


def fit_model(model_obj, params=cfg.FIT_PARAMS, progress=True, **kwargs):
    """Optimize a model object to minimise its chi-squared (note that this
    mutates model_obj.params and changes the cached component arrays)
    """
    tuples = [(k, v) for k in params.keys() for v in params[k] if k is not 'spiral']
    tuples += [
        (f'spiral{i}', v)
        for i in range(len(model_obj.spiral_distances))
        for v in params['spiral']
    ]
    p0 = model_obj.params[tuples].dropna()
    bounds = [cfg.PARAM_BOUNDS[param] for param in p0.index.levels[1][p0.index.codes[1]]]
    def _func(p):
        new_params = pd.Series(p, index=p0.index)
        r = model_obj.render(params=new_params)
        cq = chisq(r, model_obj.data, model_obj.sigma_image)
        if np.isnan(cq):
            return 1E5
        return cq
    print(f'Optimizing {len(p0)} parameters')
    print(f'Original chisq: {_func(p0.values):.4f}')
    print()
    if progress:
        with tqdm(desc='Fitting model', leave=True) as pbar:
            pbar.set_description(f'chisq={_func(p0):.4f}')
            def update_bar(*args):
                pbar.update(1)
                pbar.set_description(f'chisq={_func(args[0]):.4f}')
            res = minimize(_func, p0, bounds=bounds, callback=update_bar, **kwargs)
    else:
        res = minimize(_func, p0, bounds=bounds, **kwargs)
    print(f'Final chisq: {res["fun"]:.4f}')
    final_params = model_obj.params.copy()
    final_params[p0.index] = res['x']
    tuned_model = model_obj.to_dict(params=final_params)
    return res, tuned_model
