import numpy as np
import scipy.stats as st


def negative_log_likelihood(
    render, target, sigma,
    params=None, initial_params=None, param_sigma=None
):
    # sum of likelihood from data and likelihood from priors
    scaled_diff = (target - render) / sigma
    render_nnlf = st.norm.nnlf((0, 1), scaled_diff.ravel())
    if param_sigma is None:
        return render_nnlf
    params_nnlf = st.norm.nnlf((0, 1), np.asarray([
        (params[i] - initial_params[i]) / param_sigma[i]
        for i in param_sigma.index
        if np.isfinite(param_sigma[i])
    ]))
    return render_nnlf + params_nnlf
