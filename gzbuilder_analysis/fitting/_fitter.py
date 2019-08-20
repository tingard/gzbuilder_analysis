import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize, dual_annealing
from sklearn.metrics import mean_squared_error
from gzbuilder_analysis.config import PARAM_BOUNDS
from gzbuilder_analysis.parsing import sanitize_model


def get_bounds(template):
    bounds = [PARAM_BOUNDS[k[-1]] for k in template]
    assert len(bounds) == len(template)
    return bounds


def loss(rendered_model, galaxy_data, pixel_mask=None,
         metric=mean_squared_error):
    if pixel_mask is None:
        pixel_mask = np.ones_like(rendered_model)
    return metric(
        (rendered_model * pixel_mask).flatten(),
        0.8 * (galaxy_data * pixel_mask).flatten()
    )


def fit(model, template=None, bounds=None, progress=True, fit_kwargs={},
        anneal=False):
    """Accepts a Model or CachedModel and a parameter template list, and
    performs fiting
    """
    if template is None:
        template = model._template
    else:
        template = model.sanitize_template(template)
    if bounds is None:
        bounds = get_bounds(template)
    if anneal:
        # annealing cannot have infinite bounds
        bounds = [(max(-1E3, min(i, 1E3)), max(-1E3, min(j, 1E3)))
                  for i, j in bounds]
    if len(template) == 0:
        return model._model, dict(success=True, message='No parameters to fit')
    p0 = model.to_p(template=template)

    assert len(p0) == len(bounds)

    def f(p):
        m = model.from_p(p, template=template)
        try:
            try:
                r = model.cached_render(m)
            except AttributeError:
                r = model.render(m)
        except ZeroDivisionError:
            return 1E5
        r = np.clip(r, -1E7, 1E7)
        return loss(r, model.data, pixel_mask=model.pixel_mask)

    if progress:
        with tqdm(desc='Fitting model', leave=False) as pbar:
            cb = fit_kwargs.pop('callback', lambda *p: None)

            def update_bar(*args):
                if len(args) == 3:
                    pbar.desc
                pbar.update(1)
                cb(*args)
            if anneal:
                res = dual_annealing(f, bounds=bounds, callback=update_bar,
                                     **fit_kwargs)
            else:
                res = minimize(f, p0, bounds=bounds, callback=update_bar,
                               **fit_kwargs)
    else:
        if anneal:
            res = dual_annealing(f, bounds=bounds, **fit_kwargs)
        else:
            res = minimize(f, p0, bounds=bounds, **fit_kwargs)
    new_model = sanitize_model(
        model.from_p(res['x'], template=template)
    )
    return new_model, res
