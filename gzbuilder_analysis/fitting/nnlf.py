import jax.numpy as np
from jax import jit


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
