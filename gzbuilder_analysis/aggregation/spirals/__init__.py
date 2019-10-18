import numpy as np
import pandas as pd
from scipy.optimize import minimize
from shapely.geometry import LineString
from scipy.interpolate import splprep, splev, interp1d
from scipy import integrate
import json


def r_theta_from_xy(x, y, mux=0, muy=0):
    return (
        np.sqrt((x - mux)**2 + (y - muy)**2),
        np.arctan2((y - muy), (x - mux))
    )


def xy_from_r_theta(r, theta, mux=0, muy=0):
    return np.stack((mux + r * np.cos(theta), muy + r * np.sin(theta)))


def get_drawn_arms(models, min_n=5):
    """Extract the drawn spiral arms from a Series of models, removing
    self-intersecting lines and lines with fewer than min_n points
    """
    tuples = [
        (i, j)
        for i in models.index
        for j in range(len(models.loc[i]['spiral']))
    ]
    idx = pd.MultiIndex.from_tuples(tuples, names=['model_index', 'arm_index'])
    return pd.Series([
        points
        if len(points) > min_n and LineString(points).is_simple
        else np.nan
        for _,  model in models.iteritems()
        for i, (points, params) in enumerate(model['spiral'])
    ], index=idx).dropna()


def split_arms_at_center(arms, image_size=512, threshold=10):
    out = []
    for arm in arms:
        distances_from_centre = np.sqrt(np.add.reduce(
            (arm - [image_size / 2, image_size / 2])**2,
            axis=1
        ))
        mask = distances_from_centre < threshold
        if not np.any(mask):
            out.append(arm)
            continue
        split_arm = [
            l for l in
            (
                j if i == 0 else j[1:]
                for i, j in enumerate(np.split(arm, np.where(mask)[0]))
            )
            if len(l) > 1
        ]
        out.extend(split_arm)
    return out


def equalize_arm_length(arms, method=np.max):
    u_new = np.linspace(0, 1, method([len(i) for i in arms]))
    return [
        np.array(splev(u_new, splprep(arm.T, s=0, k=1)[0])).T
        for arm in arms
    ]


def weight_r_by_n_arms(R, groups):
    radiuses = [R[groups == g] for g in np.unique(groups)]

    r_bins = np.linspace(np.min(R), np.max(R), 100)
    counts = np.zeros(r_bins.shape)
    for i, _ in enumerate(r_bins[:-1]):
        n = sum(
            1
            for r in radiuses
            if np.any(r >= r_bins[i]) and np.any(r <= r_bins[i+1])
        )
        counts[i] = max(0, n)
    return interp1d(r_bins, counts/sum(counts))


def get_sample_weight(R, groups, bar_length=0):
    w = np.ones(R.shape[0])
    w *= R**2
    w *= weight_r_by_n_arms(R, groups)(R)
    w[R < bar_length] = 0
    w /= np.average(w)
    return w


def get_pitch_angle(b, sigma_b):
    # ch is true if spiral is clockwise
    pa = np.rad2deg(np.arctan(b))
    chirality = np.sign(pa)
    sigma_pa = np.rad2deg(np.sqrt(sigma_b**2 / (b**2 + 1)**2))
    return np.abs(pa), sigma_pa, chirality


def theta_from_pa(r, pa, C=0, return_err=False, **kwargs):
    # numerical integration of dt/dr = 1 / (r * tan(pa))
    assert np.all(pa > -90) and np.all(pa < 90)
    assert np.all(r > 0)
    # set default integration convergence tolerance
    kwargs.setdefault('epsabs', 1E-5)
    # create a function to interpolate the pitch angle with radius
    pa_wrt_r = interp1d(
        r,
        np.deg2rad(pa),
        bounds_error=False,
        fill_value='extrapolate'
    )

    # define the function to be integrated
    def f(r):
        return 1 / (r * np.tan(pa_wrt_r(r)))

    th = np.zeros(len(r))
    th_err = np.zeros(len(r))
    for i in range(1, len(r)):
        step = integrate.quad(f, r[0], r[i], **kwargs)
        th[i] = step[0]
        th_err[i] = step[1]
    if return_err:
        return th + C, th_err
    return th + C


def fit_varying_pa(arm, r, pa):
    th = theta_from_pa(r, pa)

    def f(p):
        v = interp1d(
            th + p[0], r,
            bounds_error=False,
            fill_value='extrapolate'
        )
        return abs(arm.R - v(arm.t)).sum()

    res = minimize(f, np.array((0,)))
    return th + res['x'][0]


def pa_from_r_theta(r, th):
    return np.rad2deg(np.arctan(
        np.gradient(np.log(r), th)
    ))


# def deproject_drawn_arms(drawn_arms, phi, ba):
#     """Correct for a galaxy's inclination by deprojecting its ellipticity back
#     to a circle
#     """
#     pass
#
#
# def split_arms_at_centre(drawn_arms, centre_size):
#     """Split arms and remove points within a central radius
#     """
#     pass
#
#
# def cluster_spirals(drawn_arms, image_size=(512, 512), centre_size=0.02):
#     """Cluster an array of drawn arms and return the points in each cluster,
#     splitting arms and removing points at the centre
#     """
#     pass
#
#
# def fit_log_spirals(point_cloud, clean=True):
#     pass
