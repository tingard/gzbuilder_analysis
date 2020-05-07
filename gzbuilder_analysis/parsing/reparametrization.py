import re
from copy import deepcopy
from functools import reduce
import pandas as pd
import jax.numpy as np
from ..rendering.sersic import sersic_ltot, sersic_I
from ..rendering.spiral import spiral_arm
from ..rendering import get_spirals
from gzbuilder_analysis.config import COMPONENT_PARAM_BOUNDS
from gzbuilder_analysis import df_to_dict


EMPTY_SERSIC = pd.Series(
    dict(mux=np.nan, muy=np.nan, Re=0.5, roll=0, q=1, I=0, n=1, c=2)
)
EMPTY_SERSIC_ERR = pd.Series(
    dict(mux=np.nan, muy=np.nan, Re=0.1, roll=0.01, q=0.01, I=0.01, n=0.01, c=0.01)
)


def get_centre(model):
    centre_mux = np.nanmean(np.array([
        model.get(k, {}).get('mux', np.nan) for k in ('bulge', 'bar')
        if model.get(k, None) is not None
    ]))
    centre_muy = np.nanmean(np.array([
        model.get(k, {}).get('muy', np.nan) for k in ('bulge', 'bar')
        if model.get(k, None) is not None
    ]))
    return dict(mux=float(centre_mux), muy=float(centre_muy))


# disk_spiral_L = (
#     final_model[('disk', 'L')]
#     + (comps['spiral'].sum() if 'spiral' in comps else 0)
# )
# # fractions were originally parametrized vs the disk and spirals (bulge
# # had no knowledge of bar and vice versa)
# bulge_frac = final_model.get(('bulge', 'frac'), 0)
# bar_frac = final_model.get(('bar', 'frac'), 0)
#
# bulge_L = bulge_frac * disk_spiral_L / (1 - bulge_frac)
# bar_L = bar_frac * disk_spiral_L / (1 - bar_frac)
# gal_L = disk_spiral_L + bulge_L + bar_L
#
# bulge_frac = bulge_L / (disk_spiral_L + bulge_L + bar_L)
# bar_frac = bar_L / (disk_spiral_L + bulge_L + bar_L)
#
# deparametrized_model = from_reparametrization(final_model, o)


def to_reparametrization(model, image_size):
    """Reparametrize a Galaxy Builder model to the parametrization used for
    fitting

    Steps involved:
    - Extract position of bulge and bar components to one galaxy "centre",
      using the mean values
    - Change bulge and bar effective radius to be relative to the disks
      (renamed scale)
    - Change the disk half-light intensity to be its total luminosity
    - Change the bulge and bar half-light intensity I to be the luminosity
      relative to the disk (named frac)
    """
    n_arms = sum(1 for i in model['spiral'].index if 'I.' in i)
    has_bulge = model.get('bulge', None) is not None
    has_bar = model.get('bar', None) is not None

    disk_L = float(sersic_ltot(
        model['disk']['I'],
        model['disk']['Re'],
        model['disk']['q'],
        n=1, c=2,
    ))
    bulge_L = float(sersic_ltot(
        model['bulge']['I'],
        model['bulge']['Re'],
        model['bulge']['q'],
        n=model['bulge']['n'],
        c=2
    )) if has_bulge else 0
    bar_L = float(sersic_ltot(
        model['bar']['I'],
        model['bar']['Re'],
        model['bar']['q'],
        model['bar']['n'],
        model['bar']['c']
    )) if has_bar else 0
    spiral_points = get_spirals(model.to_dict(), 2, model[('disk', 'roll')])
    spiral_params = (
        {
            re.sub(r'\.[0-9]+', '', k): v
            for k, v in model['spiral'].items()
            if '.{}'.format(i) in k
        }
        for i in range(n_arms)
    )
    spiral_L = sum(
        spiral_arm(
            arm_points=points,
            params=params,
            disk=model['disk'],
            image_size=image_size,
        ).sum()
        for points, params in zip(spiral_points, spiral_params)
    )
    total_L = disk_L + spiral_L + bulge_L + bar_L
    bulge_frac = bulge_L / total_L
    bar_frac = bar_L / total_L
    return pd.DataFrame(dict(
        disk=dict(
            mux=model['disk']['mux'],
            muy=model['disk']['muy'],
            q=model['disk']['q'],
            roll=model['disk']['roll'],
            Re=model['disk']['Re'],
            L=disk_L,
        ),
        bulge=dict(
            q=model['bulge']['q'],
            roll=model['bulge']['roll'],
            scale=model['bulge']['Re'] / model['disk']['Re'],
            frac=bulge_frac,
            n=model['bulge']['n'],
        ) if has_bulge else None,
        bar=dict(
            q=model['bar']['q'],
            roll=model['bar']['roll'],
            scale=model['bar']['Re'] / model['disk']['Re'],
            frac=bar_frac,
            n=model['bar']['n'],
            c=model['bar']['c']
        ) if has_bar else None,
        centre=(
            get_centre(model)
            if has_bulge or has_bar
            else None
        ),
        spiral=deepcopy(model['spiral'])
    )).unstack().dropna()


def from_reparametrization(model, optimizer):
    """Undo the reparametrization used for fitting
    """
    model_ = model.copy()
    comps = optimizer.render_comps(model_.to_dict())
    disk_spiral_L = (
        model_[('disk', 'L')]
        + (comps['spiral'].sum() if 'spiral' in comps else 0)
    )
    model_[('disk', 'I')] = sersic_I(
        model_[('disk', 'L')],
        model_[('disk', 'Re')],
        model_[('disk', 'q')],
    )
    model_[('disk', 'L')] = np.nan
    if 'bulge' in model_:
        bulge_L = (
            model_[('bulge', 'frac')] * (disk_spiral_L)
            / (1 - model_[('bulge', 'frac')])
        )
        bulge_Re = model_[('disk', 'Re')] * model_[('bulge', 'scale')]
        bulge_I = sersic_I(
            bulge_L,
            bulge_Re,
            model_[('bulge', 'q')],
            model_[('bulge', 'n')],
        )
        model_[('bulge', 'mux')] = model_[('centre', 'mux')]
        model_[('bulge', 'muy')] = model_[('centre', 'muy')]
        model_[('bulge', 'Re')] = bulge_Re
        model_[('bulge', 'I')] = bulge_I
        model_[[('bulge', 'scale'), ('bulge', 'frac')]] = np.nan

    if 'bar' in model_:
        bar_L = (
            model_[('bar', 'frac')] * (disk_spiral_L)
            / (1 - model_[('bar', 'frac')])
        )
        bar_Re = model_[('disk', 'Re')] * model_[('bar', 'scale')]
        bar_I = sersic_I(
            bar_L,
            bar_Re,
            model[('bar', 'q')],
            model[('bar', 'n')],
            model[('bar', 'c')],
        )
        model_[('bar', 'mux')] = model_[('centre', 'mux')]
        model_[('bar', 'muy')] = model_[('centre', 'muy')]
        model_[('bar', 'Re')] = bar_Re
        model_[('bar', 'I')] = bar_I
        model_[[('bar', 'scale'), ('bar', 'frac')]] = np.nan

    if 'centre' in model:
        model_['centre'] = np.nan
    return model_.dropna().sort_index(level=0)


def get_reparametrized_errors(agg_res):
    disk = agg_res.model['disk']
    if 'bulge' in agg_res.model:
        bulge = agg_res.model['bulge']
    else:
        bulge = EMPTY_SERSIC
    if 'bar' in agg_res.model:
        bar = agg_res.model['bar']
    else:
        bar = EMPTY_SERSIC

    disk_e = agg_res.errors['disk']
    if 'bulge' in agg_res.errors:
        bulge_e = agg_res.errors['bulge']
    else:
        bulge_e = EMPTY_SERSIC_ERR
    if 'bar' in agg_res.errors:
        bar_e = agg_res.errors['bar']
    else:
        bar_e = EMPTY_SERSIC_ERR

    errs = pd.DataFrame(
        [],
        columns=['disk', 'bulge', 'bar', 'spiral'],
        dtype=np.float64
    )
    errs['disk'] = disk_e.copy()
    # it is possible that we have zero error for ellipticity, which will
    # cause problems. Instead, fix it as a small value

    errs.loc['q', 'disk'] = max(0.001, errs.loc['q', 'disk'])
    errs.loc['L', 'disk'] = np.inf
    errs.loc['I', 'disk'] = np.nan
    errs.loc['n', 'disk'] = np.nan
    errs.loc['c', 'disk'] = np.nan

    errs['bulge'] = bulge_e.copy()
    errs.loc['q', 'bulge'] = max(0.001, errs.loc['q', 'bulge'])
    errs.loc['scale', 'bulge'] = bulge.Re / disk.Re * np.sqrt(
        bulge_e.Re**2 / bulge.Re**2
        + disk_e.Re**2 / disk.Re**2
    )
    errs.loc['frac', 'bulge'] = np.inf
    errs.loc['I', 'bulge'] = np.nan
    errs.loc['Re', 'bulge'] = np.nan
    errs.loc['c', 'bulge'] = np.nan

    errs['bar'] = bar_e.copy()
    errs.loc['q', 'bar'] = max(0.001, errs.loc['q', 'bar'])
    errs.loc['scale', 'bar'] = bar.Re / disk.Re * np.sqrt(
        bar_e.Re**2 / bar.Re**2
        + disk_e.Re**2 / disk.Re**2
    )
    errs.loc['frac', 'bar'] = np.inf
    errs.loc['I', 'bar'] = np.nan
    errs.loc['Re', 'bar'] = np.nan

    errs.loc['mux', 'centre'] = np.sqrt(np.nansum(
        np.array((disk_e.mux**2, bulge_e.mux**2))
    ))
    errs.loc['muy', 'centre'] = np.sqrt(np.nansum(
        np.array((disk_e.muy**2, bulge_e.muy**2))
    ))

    for i in range(len(agg_res.spiral_arms)):
        errs.loc['I.{}'.format(i), 'spiral'] = np.inf
        errs.loc['falloff.{}'.format(i), 'spiral'] = np.inf
        errs.loc['spread.{}'.format(i), 'spiral'] = np.inf
        errs.loc['A.{}'.format(i), 'spiral'] = 0.01
        errs.loc['phi.{}'.format(i), 'spiral'] = 1
        errs.loc['t_min.{}'.format(i), 'spiral'] = np.deg2rad(0.5)
        errs.loc['t_max.{}'.format(i), 'spiral'] = np.deg2rad(0.5)
    return df_to_dict(errs)


def get_limits(agg_res):
    n_spirals = len(agg_res.spiral_arms)
    return {
        'disk': {
            'L': COMPONENT_PARAM_BOUNDS['disk']['L'],
            'mux': COMPONENT_PARAM_BOUNDS['disk']['mux'],
            'muy': COMPONENT_PARAM_BOUNDS['disk']['muy'],
            'q': COMPONENT_PARAM_BOUNDS['disk']['q'],
            'roll': COMPONENT_PARAM_BOUNDS['disk']['roll'],
            'Re': COMPONENT_PARAM_BOUNDS['disk']['Re'],
        },
        'bulge': {
            'frac': COMPONENT_PARAM_BOUNDS['bulge']['frac'],
            'mux': COMPONENT_PARAM_BOUNDS['bulge']['mux'],
            'muy': COMPONENT_PARAM_BOUNDS['bulge']['muy'],
            'n': COMPONENT_PARAM_BOUNDS['bulge']['n'],
            'q': COMPONENT_PARAM_BOUNDS['bulge']['q'],
            'roll': COMPONENT_PARAM_BOUNDS['bulge']['roll'],
            'scale': COMPONENT_PARAM_BOUNDS['bulge']['scale'],
        },
        'bar': {
            'c': COMPONENT_PARAM_BOUNDS['bar']['c'],
            'frac': COMPONENT_PARAM_BOUNDS['bar']['frac'],
            'mux': COMPONENT_PARAM_BOUNDS['bar']['mux'],
            'muy': COMPONENT_PARAM_BOUNDS['bar']['muy'],
            'n': COMPONENT_PARAM_BOUNDS['bar']['n'],
            'q': COMPONENT_PARAM_BOUNDS['bar']['q'],
            'roll': COMPONENT_PARAM_BOUNDS['bar']['roll'],
            'scale': COMPONENT_PARAM_BOUNDS['bar']['scale'],
        },
        'spiral': reduce(lambda a, b: {**a, **b}, ({
            'I.{}'.format(i): COMPONENT_PARAM_BOUNDS['spiral']['I'],
            'A.{}'.format(i): COMPONENT_PARAM_BOUNDS['spiral']['A'],
            'falloff.{}'.format(i): COMPONENT_PARAM_BOUNDS['spiral']['falloff'],
            'phi.{}'.format(i): COMPONENT_PARAM_BOUNDS['spiral']['phi'],
            'spread.{}'.format(i): COMPONENT_PARAM_BOUNDS['spiral']['spread'],
            't_min.{}'.format(i): COMPONENT_PARAM_BOUNDS['spiral']['t_min'],
            't_max.{}'.format(i): COMPONENT_PARAM_BOUNDS['spiral']['t_max'],
        } for i in range(n_spirals))) if n_spirals > 0 else {},
        'centre': COMPONENT_PARAM_BOUNDS['centre'],
    }
