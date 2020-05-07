import numpy as np
import pandas as pd
from ..config import COMPONENT_CLUSTERING_PARAMS
from .__geom_prep import make_ellipse, make_box
from .spirals import get_drawn_arms
from .__cluster import cluster_components
from .__aggregate import circular_error, aggregate_components
from .__aggregation_result import AggregationResult


def get_geoms(model):
    """Function to obtain shapely geometries from parsed Zooniverse
    classifications
    """
    out = dict()
    for k, f in zip(
        ('disk', 'bulge', 'bar'),
        (make_ellipse, make_ellipse, make_box)
    ):
        try:
            out[k] = f(model[k])
        except KeyError:
            pass
    return out


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
