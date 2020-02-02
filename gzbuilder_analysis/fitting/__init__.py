# import numpy as np
# import pandas as pd
# from .__model import Model
# from .__nnlf import negative_log_likelihood
# import gzbuilder_analysis.config as cfg
# from scipy.optimize import minimize
# from tqdm import tqdm


# def chisq(model_data, galaxy_data, sigma_image, pixel_mask=None):
#     # chisq = 1/N_dof * sum(f_data(x, y) - f_model(x, y))^2 / sigma(x, y)^2
#     # Assume N_dof ~ number of unmasked pixels
#     # pixel_mask values of True means a pixel should be masked
#     if type(galaxy_data) != np.ma.core.MaskedArray:
#         galaxy_data = np.ma.masked_array(galaxy_data, pixel_mask)
#     return 1 / len(galaxy_data.compressed()) * np.sum(
#         np.clip((model_data - galaxy_data) / sigma_image, -1E5, 1E5)
#         .astype(np.float64)**2
#     )
#
#
# def fit_model(model_obj, params=cfg.FIT_PARAMS, progress=True, **kwargs):
#     """Optimize a model object to minimise its chi-squared (note that this
#     mutates model_obj.params and changes the cached component arrays)
#     """
#     tuples = [(k, v) for k in params.keys() for v in params[k] if k is not 'spiral']
#     tuples += [
#         (f'spiral{i}', v)
#         for i in range(len(model_obj.spiral_distances))
#         for v in params['spiral']
#     ]
#     p0 = model_obj.params[tuples].dropna()
#     if len(p0) == 0:
#         print('No parameters to optimize')
#         return {}, model_obj.to_dict()
#     bounds = [cfg.PARAM_BOUNDS[param] for param in p0.index.levels[1][p0.index.codes[1]]]
#
#     def _func(p):
#         new_params = pd.Series(p, index=p0.index)
#         r = model_obj.render(params=new_params)
#         cq = chisq(r, model_obj.data, model_obj.sigma_image)
#         if np.isnan(cq):
#             return 1E5
#         return cq
#
#     print(f'Optimizing {len(p0)} parameters')
#     print(f'Original chisq: {_func(p0.values):.4f}')
#     print()
#     if progress:
#         with tqdm(desc='Fitting model', leave=True) as pbar:
#             pbar.set_description(f'chisq={_func(p0):.4f}')
#
#             def update_bar(*args):
#                 pbar.update(1)
#                 pbar.set_description(f'chisq={_func(args[0]):.4f}')
#             res = minimize(_func, p0, bounds=bounds, callback=update_bar, **kwargs)
#     else:
#         res = minimize(_func, p0, bounds=bounds, **kwargs)
#     print(f'Final chisq: {res["fun"]:.4f}')
#     # the fitting process allows parameters to vary outside conventional bounds
#     # (roll > 2*pi, axis ratio > 1 etc...). We fix this before exiting
#     model_obj.sanitize()
#
#     # obtain a model dict to return:
#     final_params = model_obj.params.copy()
#     final_params[p0.index] = res['x']
#     # we use the model object rather than `from_pandas` in order to recover the
#     # original spiral arm points, which would otherwise have been lost
#     tuned_model = model_obj.to_dict(params=final_params)
#     return res, tuned_model
