import numpy as np
from scipy.interpolate import splprep, splev, interp1d
import pandas as pd
from shapely.geometry import LineString


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
    # We increase the weights of arms where we have multiple overlapping drawn poly-lines,
    # and also with radius (to represent the decreasing uncertainty in theta)
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


def pa_from_r_theta(r, th):
    return np.rad2deg(np.arctan(
        np.gradient(np.log(r), th)
    ))
