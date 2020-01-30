import numpy as np
import pandas as pd
from scipy.optimize import minimize
from gzbuilder_analysis.config import DEFAULT_SPIRAL
from gzbuilder_analysis.parsing import downsample
from gzbuilder_analysis.parsing import sanitize_param_dict
from .__geom_prep import ellipse_from_param_list, box_from_param_list, \
    get_param_dict, get_param_list, make_ellipse, make_box
from .__jaccard import jaccard_distance


def aggregate_components(clustered_models):
    """Accepts clusters of components and constructs an aggregate model
    """
    disk_cluster_geoms = clustered_models['disk'].apply(make_ellipse)
    bulge_cluster_geoms = clustered_models['bulge'].apply(make_ellipse)
    bar_cluster_geoms = clustered_models['bar'].apply(make_box)
    # calculate an aggregate disk
    if len(disk_cluster_geoms) != 0:
        aggregate_disk = aggregate_model_clusters_mean(
            clustered_models['disk']
        )
        aggregate_disk = {
            **aggregate_geom_jaccard(
                disk_cluster_geoms.values,
                x0=get_param_list(aggregate_disk)
            ),
            'I': 0.2,
            'n': 1.0,
            'c': 2.0,
        }
    else:
        aggregate_disk = None

    # calculate an aggregate bulge
    if len(bulge_cluster_geoms) != 0:
        aggregate_bulge = aggregate_model_clusters_mean(
            clustered_models['bulge']
        )
        aggregate_bulge = {
            **aggregate_geom_jaccard(
                bulge_cluster_geoms.values,
                x0=get_param_list(aggregate_bulge)
            ),
            'I': 0.01,
            'n': 1.0,
            'c': 2.0,
        }
    else:
        aggregate_bulge = None

    # calculate an aggregate bar
    if len(bar_cluster_geoms) != 0:
        aggregate_bar = aggregate_model_clusters_mean(
            clustered_models['bar']
        )
        aggregate_bar = {
            **aggregate_geom_jaccard(
                bar_cluster_geoms.values,
                x0=get_param_list(aggregate_bar),
                constructor_func=box_from_param_list,
            ),
            'I': 0.01,
            'n': 0.5,
            'c': 2,
        }
    else:
        aggregate_bar = None

    agg_model = dict(
        disk=aggregate_disk,
        bulge=aggregate_bulge,
        bar=aggregate_bar,
    )
    try:
        agg_model['spiral'] = [
            (downsample(a.reprojected_log_spiral), DEFAULT_SPIRAL)
            for a in clustered_models['spiral'].get_arms()
        ]
    except KeyError:
        agg_model['spiral'] = None
    return agg_model


def aggregate_model_clusters_mean(component_clusters):
    """naively take the mean of each parameter of each cluster of components,
    as this is mainly to choose the slider values
    """
    df = component_clusters.apply(pd.Series)
    mean_component = df.mean().to_dict()
    mean_component['roll'] = df['roll'].mean()
    return mean_component


def aggregate_geom_jaccard(geoms, x0=np.array((256, 256, 5, 0.7, 0)),
                           constructor_func=ellipse_from_param_list):
    def __distance_func(p):
        p = np.array(p)
        assert len(p) == 5, 'Invalid number of parameters supplied'
        comp = constructor_func(p)
        s = sum(jaccard_distance(comp, other) for other in geoms)
        return s
    # sanitze results rather than imposing bounds to avoid getting stuck in
    # local minima
    return sanitize_param_dict(
        get_param_dict(
            minimize(__distance_func, x0)['x']
        )
    )
