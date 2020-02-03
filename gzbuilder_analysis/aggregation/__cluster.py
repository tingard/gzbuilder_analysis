import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from shapely.affinity import scale as shapely_scale
from gzbuilder_analysis.config import COMPONENT_CLUSTERING_PARAMS
from gzbuilder_analysis.parsing import parse_annotation
from .__geom_prep import make_ellipse, make_box
from .jaccard import make_jaccard_distances
from .spirals import get_drawn_arms
from .spirals.oo import Pipeline


def cluster_components(models=None, classifications=None, image_size=(512, 512),
                       phi=None, ba=None, params=COMPONENT_CLUSTERING_PARAMS,
                       warn=True):
    """Accepts a list of models (or classifications) and returns clusters of
    disks, bulges, bars and spiral arms
    """
    if classifications is not None and image_size is not None:
        models = classifications['annotations'].apply(json.loads)\
            .apply(parse_annotation, image_size=image_size, ignore_scale=True)
    elif models is not None:
        models = models
    else:
        raise ValueError('Must provide models or classifications for clustering')
    models_df = models.apply(pd.Series)
    components = {
        k: models_df[k]
        for k in ('disk', 'bulge', 'bar')
    }
    # we size-up the geometries by a factor of 3 for clustering, such that the Jaccard
    # distances are not very high for all of them
    geoms = {
        'disk': components['disk'].apply(
            make_ellipse
        ).dropna(),
        'bulge': components['bulge'].apply(
            make_ellipse
        ).dropna(),
        'bar': components['bar'].apply(
            make_box
        ).dropna(),
    }
    clusters = {
        k: components[k].loc[cluster_geoms(
            geoms[k],
            params[k]['eps'],
            params[k]['min_samples'],
        ).index]
        for k in ('disk', 'bulge', 'bar')
    }
    if phi is None or ba is None:
        if warn:
            print('No spiral inclination and position angle provided, ignoring spirals.')
            print(
                'Please re-run spiral clustering with the position and ellipticity',
                'obtained from aggregation of disk'
                )
        return clusters
    drawn_arms = get_drawn_arms(models, min_n=5)

    clusters['spiral'] = Pipeline(
        drawn_arms.values,
        phi=phi, ba=ba,
        image_size=image_size
    )
    return clusters


def cluster_geoms(geoms, eps, min_samples):
    """accepts a Series of geometries and returns only the geometries in the
    largest cluster (preserving Index)
    """
    labels = get_cluster_labels(geoms, eps, min_samples)
    if len(labels[labels >= 0].dropna()) == 0:
        return geoms.loc[[]]
    largest_label = labels[labels >= 0].dropna().groupby(labels).count().idxmax()
    return geoms[labels == largest_label]


def get_cluster_labels(geoms, eps, min_samples):
    filtered = geoms.dropna()
    if len(filtered) == 0:
        return pd.Series({i: np.nan for i in geoms.index})
    distances = make_jaccard_distances(filtered)
    clf = DBSCAN(eps=eps, min_samples=min_samples,
                 metric='precomputed')
    clf.fit(distances)
    labels = pd.Series(np.full(len(geoms), np.nan), index=geoms.index)
    labels[filtered.index] = clf.labels_
    return labels
