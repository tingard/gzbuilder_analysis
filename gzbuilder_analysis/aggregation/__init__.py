import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from gzbuilder_analysis.spirals import get_drawn_arms
from gzbuilder_analysis.spirals.oo import Pipeline
import gzbuilder_analysis.parsing as pa
from gzbuilder_analysis.aggregation import average_shape_helpers as ash
from gzbuilder_analysis.config import COMPONENT_CLUSTERING_PARAMS
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)


def sanitize_model(model):
    try:
        bar_axratio = model.get('bar', {}).get('axRatio', 1)
        if bar_axratio > COMPONENT_CLUSTERING_PARAMS['max_bar_axratio']:
            model['bar'] = None
    except AttributeError:
        pass
    return model


def get_geoms(model_details):
    """Function to obtain shapely geometries from parsed Zooniverse
    classifications
    """
    disk = ash.make_ellipse(model_details['disk'])
    bulge = ash.make_ellipse(model_details['bulge'])
    bar = ash.make_box(model_details['bar'])
    return disk, bulge, bar


def cluster_components(geoms):
    disk_labels = ash.cluster_comp(
        geoms['disk'],
        COMPONENT_CLUSTERING_PARAMS['disk']['eps'],
        COMPONENT_CLUSTERING_PARAMS['disk']['min_samples'],
    )
    bulge_labels = ash.cluster_comp(
        geoms['bulge'],
        COMPONENT_CLUSTERING_PARAMS['bulge']['eps'],
        COMPONENT_CLUSTERING_PARAMS['bulge']['min_samples'],
    )
    bar_labels = ash.cluster_comp(
        geoms['bar'],
        COMPONENT_CLUSTERING_PARAMS['bar']['eps'],
        COMPONENT_CLUSTERING_PARAMS['bar']['min_samples'],
    )
    return disk_labels, bulge_labels, bar_labels


def aggregate_comp_mean(comps):
    comps = list(map(ash.sanitize_param_dict, comps))
    out = {'i0': 1, 'n': 1, 'c': 2}
    if len(comps) == 0:
        return out
    keys = list(comps[0].keys())
    for key in keys:
        if key == 'roll':
            clustered_angles = [
                np.rad2deg(i['roll'])
                for i in comps
            ]
            out['roll'] = np.deg2rad(np.mean(clustered_angles))
        else:
            out[key] = np.mean([i.get(key, np.nan) for i in comps], axis=0)
    return out


def aggregate_geom_jaccard(geoms, x0=np.array((256, 256, 5, 0.7, 0)),
                           constructor_func=ash.ellipse_from_param_list):
    def __distance_func(p):
        p = np.array(p)
        assert len(p) == 5, 'Invalid number of parameters supplied'
        comp = constructor_func(p)
        s = sum(ash.jaccard_distance(comp, other) for other in geoms)
        return s
    # sanitze results rather than imposing bounds to avoid getting stuck in
    # local minima
    return ash.sanitize_param_dict(
        ash.get_param_dict(
            minimize(__distance_func, x0)['x']
        )
    )


def get_aggregate_components(geoms, models, labels):
    cluster_labels = list(map(ash.largest_cluster_label, labels))
    cluster_masks = [a == b for a, b in zip(labels, cluster_labels)]
    disk_cluster_geoms = geoms['disk'][cluster_masks[0]]
    bulge_cluster_geoms = geoms['bulge'][cluster_masks[1]]
    bar_cluster_geoms = geoms['bar'][cluster_masks[2]]
    # calculate an aggregate disk
    if not np.any(labels[0] == cluster_labels[0]):
        aggregate_disk = None
    else:
        aggregate_disk = aggregate_comp_mean(
            models[labels[0] == cluster_labels[0]].apply(
                lambda v: v.get('disk', None)
            ).dropna()
        )
        aggregate_disk = {
            **aggregate_disk,
            **aggregate_geom_jaccard(
                disk_cluster_geoms.values,
                x0=ash.get_param_list(aggregate_disk)
            )
        }
    # calculate an aggregate bulge
    if not np.any(labels[1] == cluster_labels[1]):
        aggregate_bulge = None
    else:
        aggregate_bulge = aggregate_comp_mean(
            models[labels[1] == cluster_labels[1]].apply(
                lambda v: v.get('bulge', None)
            ).dropna()
        )
        aggregate_bulge = {
            **aggregate_bulge,
            **aggregate_geom_jaccard(
                bulge_cluster_geoms.values,
                x0=ash.get_param_list(aggregate_bulge)
            )
        }
    # calculate an aggregate bar
    if not np.any(labels[2] == cluster_labels[2]):
        aggregate_bar = None
    else:
        aggregate_bar = aggregate_comp_mean(
            models[labels[2] == cluster_labels[2]].apply(
                lambda v: v.get('bar', None)
            ).dropna()
        )
        aggregate_bar = {
            **aggregate_bar,
            **aggregate_geom_jaccard(
                bar_cluster_geoms.values,
                x0=ash.get_param_list(aggregate_bar),
                constructor_func=ash.box_from_param_list,
            )
        }
    return aggregate_disk, aggregate_bulge, aggregate_bar


def get_spiral_arms(classifications, gal, angle, parallel=True):
    drawn_arms = get_drawn_arms(classifications)
    if len(drawn_arms) == 0:
        return []
    p = Pipeline(drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                 image_size=512, parallel=parallel)
    return p.get_arms()


def make_errors(models, masks):
    comps = ('disk', 'bulge', 'bar')
    params = ('axRatio', 'rEff', 'i0', 'n', 'c')
    disks, bulges, bars = [
        pd.DataFrame(j)
        for j in (
            models[masks[i]].apply(
                lambda a: a.get(comps[i])
            ).dropna().values.tolist()
            for i in range(3)
        )
    ]
    out = {}
    try:
        out['disk'] = disks.describe().drop('roll', axis=1)\
            .loc['std'].to_dict()
    except ValueError:
        out['disk'] = {k: np.nan for k in params}
    try:
        out['bulge'] = bulges.describe().drop('roll', axis=1)\
            .loc['std'].to_dict()
    except ValueError:
        out['bulge'] = {k: np.nan for k in params}
    try:
        out['bar'] = bars.describe().drop('roll', axis=1)\
            .loc['std'].to_dict()
    except ValueError:
        out['bar'] = {k: np.nan for k in params}
    return out


def make_model(classifications, gal, angle, outfile=None, parallel=True):
    annotations = classifications['annotations'].apply(json.loads)
    models = annotations\
        .apply(ash.remove_scaling)\
        .apply(pa.parse_annotation)\
        .apply(sanitize_model)
    spirals = models.apply(lambda d: d.get('spiral', None))
    geoms = pd.DataFrame(
        models.apply(get_geoms).values.tolist(),
        columns=('disk', 'bulge', 'bar')
    )
    geoms['spirals'] = spirals
    labels = list(map(np.array, cluster_components(geoms)))
    cluster_labels = list(map(ash.largest_cluster_label, labels))
    cluster_masks = [a == b for a, b in zip(labels, cluster_labels)]
    aggregate_disk, aggregate_bulge, aggregate_bar = get_aggregate_components(
        geoms, models, labels
    )
    arms = get_spiral_arms(classifications, gal, angle, parallel=parallel)
    logsps = [arm.reprojected_log_spiral for arm in arms]
    if outfile is not None:
        with open(outfile, 'w') as f:
            json.dump(
                {
                    'disk': ash.sanitize_comp_for_json(aggregate_disk),
                    'bulge': ash.sanitize_comp_for_json(aggregate_bulge),
                    'bar': ash.sanitize_comp_for_json(aggregate_bar),
                    'spirals': [a.tolist() for a in logsps],
                },
                f,
            )
    errors = make_errors(models, cluster_masks)
    return {
        'disk': aggregate_disk,
        'bulge': aggregate_bulge,
        'bar': aggregate_bar,
        'spirals': logsps,
    }, errors, cluster_masks, arms
