import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import gzbuilder_analysis.parsing as parsing
from gzbuilder_analysis.config import COMPONENT_CLUSTERING_PARAMS
from .__geom_prep import get_drawn_arms, make_ellipse, make_box
from .__cluster import cluster_components
from .__aggregate import aggregate_components


def get_geoms(model):
    """Function to obtain shapely geometries from parsed Zooniverse
    classifications
    """
    disk = make_ellipse(model['disk'])
    bulge = make_ellipse(model['bulge'])
    bar = make_box(model['bar'])
    return dict(disk=disk, bulge=bulge, bar=bar)


def make_errors(models, masks):
    comps = ('disk', 'bulge', 'bar')
    params = ('q', 'Re', 'roll', 'I', 'n', 'c')
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


def make_model(classifications, gal, angle, outfile=None, parallel=True, **kwargs):
    annotations = classifications['annotations'].apply(json.loads)
    models = annotations\
        .apply(ash.remove_scaling)\
        .apply(parsing.parse_annotation, args=((512, 512),))\
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
    arms = get_spiral_arms(
        classifications, gal, angle,
        parallel=parallel,
        wcs_in=kwargs.get('wcs_in', None),
        wcs_out=kwargs.get('wcs_out', None),
    )
    logsps = [arm.reprojected_log_spiral for arm in arms]
    agg_model = {
        'disk': aggregate_disk,
        'bulge': aggregate_bulge,
        'bar': aggregate_bar,
        'spirals': logsps,
    }
    if outfile is not None:
        with open(outfile, 'w') as f:
            json.dump(agg_model, f)
    errors = make_errors(models, cluster_masks)
    if kwargs.get('image_size', False) and kwargs.get('size_diff', False):
        agg_model = parsing.scale_aggregate_model(
            agg_model,
            image_size=kwargs['image_size'],
            size_diff=kwargs['size_diff'],
            wcs_in=kwargs.get('wcs_in'),
            wcs_out=kwargs.get('wcs_out'),
        )
    return agg_model, errors, cluster_masks, arms
