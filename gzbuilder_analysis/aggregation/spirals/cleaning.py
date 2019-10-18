from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from gzbuilder_analysis.config import SPIRAL_LOF_KWARGS
import numpy as np


def get_grouped_data(drawn_arms):
    coords = np.array([point for arm in drawn_arms for point in arm])
    groups = np.fromiter((
        g
        for i, arm in enumerate(drawn_arms)
        for g in [i]*len(arm)
    ), dtype=int, count=len(coords))
    return coords, groups


def clean_points(point_cloud):
    clf = LocalOutlierFactor(**SPIRAL_LOF_KWARGS)
    y_pred = clf.fit_predict(point_cloud)
    mask = ((y_pred + 1) / 2).astype(bool)
    return clf, mask


def clean_arms_xy(point_cloud, groups):
    s = StandardScaler()
    clf = LocalOutlierFactor(n_jobs=-1, **SPIRAL_LOF_KWARGS)
    s.fit(point_cloud)
    X_normed, Y_normed = s.transform(point_cloud).T
    standardized_cloud = np.stack((X_normed, Y_normed), axis=1)
    res = np.ones(point_cloud.shape[0]).astype(bool)
    # for each drawn arm in this cluster
    for group in np.unique(groups):
        # get a mask for all points in this arm
        testField = groups == group
        # train on all the other arms present
        X_train = standardized_cloud[~testField]
        # test on current arm
        X_test = standardized_cloud[testField]
        clf.fit(X_train)
        # save whether each point in the arm is an outlier
        res[testField] = clf.predict(X_test) > 0
    return res


def clean_arms_polar(R, theta, groups):
    """Clean drawn arms in r, theta space"""
    alg = LocalOutlierFactor(n_jobs=-1, **SPIRAL_LOF_KWARGS)

    R_normed = R / R.std()
    t_normed = theta / theta.std()
    res = np.ones(R.shape[0]).astype(bool)

    for group in np.unique(groups):
        testField = groups != group
        X_train = np.stack(
            (R_normed[testField].reshape(-1), t_normed[testField])
        ).T
        X_test = np.stack(
            (R_normed[~testField].reshape(-1), t_normed[~testField])
        ).T
        alg.fit(X_train)
        res[~testField] = alg.predict(X_test) > 0
    return res
