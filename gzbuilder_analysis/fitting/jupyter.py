import numpy as np
import string
from tqdm import tqdm
from scipy.optimize import minimize
from gzbuilder_analysis.fitting import loss
from gzbuilder_analysis.fitting._fitter import get_bounds
from IPython.display import display, update_display, HTML


class UpdatableDisplay():
    def __init__(self, thing_to_display):
        self.DISPLAY_ID = ''.join(
            np.random.choice(
                list(string.ascii_letters + string.digits),
                30
            )
        )
        display(thing_to_display, display_id=self.DISPLAY_ID)

    def __call__(self, thing_to_display):
        update_display(thing_to_display, display_id=self.DISPLAY_ID)


def live_fit(model, template=None, bounds=None, progress=True):
    """Accepts a Model or CachedModel and a parameter template list, and
    performs fiting
    """
    display(HTML(
        'Running:<br>'
        + '<code>fitted_model, res = model_fitter.fit()</code>'
    ))
    if template is None:
        template = model._template
    if bounds is None:
        bounds = get_bounds(template)

    p0 = model.to_p(template=template)

    d = UpdatableDisplay(HTML(model._repr_html_()))

    def f(p):
        m = model.from_p(p, template=template)
        try:
            r = model.cached_render(m)
        except AttributeError:
            r = model.render(m)
        return loss(r, model.data, pixel_mask=model.pixel_mask)

    with tqdm(desc='Fitting model', leave=False) as pbar:
        def update_bar(p):
            pbar.update(1)
            m = model.from_p(p, template=template)
            d(HTML(model._repr_html_(model=m)))
        res = minimize(f, p0, bounds=bounds, callback=update_bar)

    new_model = model.from_p(res['x'], template=template)
    return new_model, res
