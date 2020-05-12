import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import median_absolute_error
from gzbuilder_analysis.aggregation.spirals import metric
from gzbuilder_analysis.config import SPIRAL_BAYESIAN_RIDGE_PRIORS
# obtained by fitting a semi-truncated Gamma distribution to spiral arm width
# slider values, after removing all values at 1 (default) and 0 and 2
# (extremes)


def unwrap(theta, r, groups):
    """Unwrap grouped poly-lines, attempting to give points at the same radius
    the same value of theta (not trivial).
    This method calculates the separation between unwrapped lines, and attempts
    to find the value of delta (in multiples of 2pi) that minimizes the
    distance to the other unwrapped lines.
    """
    out = theta.copy()
    dt = (np.arange(3) - 1) * 2 * np.pi
    r_scaling = 2 * np.pi / np.max(r)
    for i, g in enumerate(np.unique(groups)):
        t_ = np.unwrap(out[groups == g])
        if i == 0:
            out[groups == g] = t_
            continue
        coords = np.stack((t_, r[groups == g] * r_scaling), axis=0)
        poss = ((coords.T + (delta, 0)).T for delta in dt)

        extremes = [np.argmin(out[groups < g]), np.argmax(out[groups < g])]
        other = np.stack((
            out[groups < g][extremes],
            r[groups < g][extremes] * r_scaling
        ), axis=0)

        d = [metric.min_arc_distance_numpy([c.T, other.T]) for c in poss]
        out[groups == g] = t_ + dt[np.argmin(d)]
    return out


def weighted_group_cross_val(pipeline, X, y, cv, groups, weights,
                             score=median_absolute_error, lower_better=True):
    """Function that performs a weighted group K-Fold cross validation on data,
    where `pipeline` is a scikit-learn estimator or Pipeline object, and cv is
    a cross-validator.
    """
    # TODO: example usage
    scores = np.zeros(cv.get_n_splits(X, y, groups=groups))
    for i, (train, test) in enumerate(cv.split(X, y, groups=groups)):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        group_weights = weights[train] / weights[train].mean()
        pipeline.fit(X_train, y_train,
                     bayesianridge__sample_weight=group_weights)
        y_pred = pipeline.predict(
            X_test,
        )
        # use the negative median absolute error (so > is better)
        if lower_better:
            scores[i] = -score(y_pred, y_test)
        else:
            scores[i] = score(y_pred, y_test)
    return scores


# Log spiral model
def get_log_spiral_pipeline():
    """Create an sklearn pipeline to fit Y = a * exp(b * X), don't bother
    with bias as logsp is self-similar. Also not scaling numbers as it makes
    it trickier to recover fitting parameters (I think?)
    """
    names = ('polynomialfeatures', 'bayesianridge')
    steps = [
        PolynomialFeatures(
            degree=1,
            include_bias=True,
        ),
        TransformedTargetRegressor(
            regressor=BayesianRidge(
                compute_score=True,
                fit_intercept=False,
                copy_X=True,
                normalize=False,
                **SPIRAL_BAYESIAN_RIDGE_PRIORS
            ),
            func=np.log,
            inverse_func=np.exp
        )
    ]
    return Pipeline(memory=None, steps=list(zip(names, steps)))


# Polynomial model
def get_polynomial_pipeline(degree):
    """Simple sklearn pipeline to fit y = sum_{i=1}^{degree} c_i * X^i
    """
    return make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False,
        ),
        BayesianRidge(
            compute_score=True,
            fit_intercept=True,
            copy_X=True,
            normalize=False,
            **SPIRAL_BAYESIAN_RIDGE_PRIORS
        )
    )


# Swing amplification model (not using sklearn pipelines)
def _swing_amplification_dydt(r, theta, b):
    """EXPERIMENTAL: the swing-amplification spiral model for flat rotation
    curves
    """
    R = 2 * b * r
    s = np.sinh(R)
    return (
        2*np.sqrt(2) / 7 * r
        * np.sqrt(1 + R / s) / (1 - R / s)
    )


def fit_swing_amplified_spiral(theta, r):
    """EXPERIMENTAL: fit a spiral model which assumes a flat rotation curve and
    the swing amplification most amplified pitch angle prediction
    """
    def f(p):
        # p = (b, r0)
        y = odeint(_swing_amplification_dydt, p[1], theta, args=(p[0],))[:, 0]
        return np.abs(y - r).sum()

    res = minimize(f, (0.1, 0.1))
    guess_b, guess_r0 = res['x']
    r_guess = odeint(_swing_amplification_dydt, guess_r0, theta,
                     args=(guess_b,))[:, 0]
    guess_sigma = (r - r_guess).std()

    return r_guess, {'b': guess_b, 'r0': guess_r0, 'sigma': guess_sigma}
