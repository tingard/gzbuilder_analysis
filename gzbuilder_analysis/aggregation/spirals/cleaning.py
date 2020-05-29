from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from gzbuilder_analysis.config import SPIRAL_LOF_KWARGS
import numpy as np


def get_grouped_data(drawn_arms):
    """Convert a list of poly-lines to a point cloud and group map
    """
    coords = np.array([point for arm in drawn_arms for point in arm])
    groups = np.fromiter((
        g
        for i, arm in enumerate(drawn_arms)
        for g in [i]*len(arm)
    ), dtype=int, count=len(coords))
    return coords, groups


def grouped_local_outlier_factor(points, groups, **kwargs):
    """Use local outlier factor to clean grouped data by training the LOF
    dectector on points not in a group before testing on points in the group.
    Repeat for each group. Operates in cartesian coordinates.
    """
    # Standardise points
    s = StandardScaler()
    s.fit(points)
    X_normed, Y_normed = s.transform(points).T
    standardized_cloud = np.stack((X_normed, Y_normed), axis=1)
    res = np.ones(points.shape[0]).astype(bool)
    clf = LocalOutlierFactor(n_jobs=-1, **kwargs)
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


def clean_arms_xy(point_cloud, groups):
    """legacy wrapper on grouped_local_outlier_factor"""
    return grouped_local_outlier_factor(
        point_cloud,
        groups,
        **SPIRAL_LOF_KWARGS,
    )
