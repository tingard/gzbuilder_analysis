import jax.numpy as np
from jax import jit


# def negative_log_likelihood(
#     render, target, sigma,
#     params=None, initial_params=None, param_sigma=None
# ):
#     # sum of likelihood from data and likelihood from priors
#     scaled_diff = (target - render) / sigma
#     render_nnlf = st.norm.nnlf((0, 1), scaled_diff.ravel())
#     if param_sigma is None:
#         return render_nnlf
#     params_nnlf = st.norm.nnlf((0, 1), np.asarray([
#         (params[i] - initial_params[i]) / param_sigma[i]
#         for i in param_sigma.index
#         if np.isfinite(param_sigma[i])
#     ]))
#     return render_nnlf + params_nnlf


@jit
def negative_log_likelihood(x):
    r"""Calclate Negative log-likelihood function for standard normally
    distributed variables `x`
    $$
    l(\mu, \sigma; x_1, ..., x_n) =
        -\frac{n}{2}\ln(2\pi)
        -\frac{n}{2}\ln(\sigma^2)
        - \frac{1}{2\sigma^2}\sum_{j=1}^n(x_j - \mu)^2
    $$
    """
    n = len(x)
    log_likelihood = (
        -n / 2 * np.log(2*np.pi)
        - 1 / 2 * np.sum(x**2)
    )
    return -log_likelihood
