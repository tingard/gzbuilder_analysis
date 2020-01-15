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
