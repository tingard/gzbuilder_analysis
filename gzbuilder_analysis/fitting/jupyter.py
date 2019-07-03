import numpy as np
import string
from tqdm import tqdm
from IPython.display import display, update_display, HTML
from gzbuilder_analysis.fitting import get_param_bounds


def interactive_fitter(model_fitter):
    display(HTML(
        'Running:<br>'
        + '<code>fitted_model, res = model_fitter.fit()</code>'
    ))
    p0, p_key, bounds = model_fitter.model.construct_p(
        model_fitter.model.base_model, bounds=get_param_bounds({})
    )
    DISPLAY_ID = ''.join(
        np.random.choice(
            list(string.ascii_letters + string.digits),
            30
        )
    )

    display(
        HTML(model_fitter.model._repr_html_(model_fitter.model.base_model)),
        display_id=DISPLAY_ID
    )

    with tqdm(desc='Fitting model') as pbar:
        def update_bar(p):
            pbar.update(1)
            new_model = model_fitter.model.update_model(
                model_fitter.model.base_model,
                p
            )
            s_html = model_fitter.model._repr_html_(new_model)
            update_display(HTML(s_html), display_id=DISPLAY_ID)
        fitted_agg_nosp_model, agg_nosp_res = model_fitter.fit(
            callback=update_bar,
            options=dict(maxiter=100)
        )

    if agg_nosp_res.success:
        display(HTML(
            '<span style="color:green">Successfuly completed fit</span>'
        ))
    else:
        display(HTML(
            '<span style="color:red">Did not fit successfully!</span>'
        ))
    return fitted_agg_nosp_model, agg_nosp_res
