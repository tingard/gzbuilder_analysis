"""Usage: (note this does not correctly update log spirals when the disk updates)
# we have target, sigma, mask, model, model_sigma
from tqdm import tqdm
from scipy.optimize import minimize
from gzbuilder_analysis.rendering.jax.spiral import vmap_polyline_distance

model = restructure_model(model)
points = model['spiral'].pop('points')

n = 5 # oversampling factor
shape = target.shape

x = np.arange(shape[1], dtype=np.float64)
y = np.arange(shape[0], dtype=np.float64)
cx, cy = np.meshgrid(x, y)
x_super = np.linspace(0.5 / n - 0.5, shape[1] - 0.5 - 0.5 / n, shape[1] * n)
y_super = np.linspace(0.5 / n - 0.5, shape[0] - 0.5 - 0.5 / n, shape[0] * n)
cx_super, cy_super = np.meshgrid(x_super, y_super)

distances = np.stack([
    vmap_polyline_distance(p, cx, cy)
    for p in points
], axis=-1)

orignal_p = pd.DataFrame(model).unstack().dropna()
p0 = to_1d(orignal_p)

args = (p0, cx_super, cy_super, distances, psf, target, sigma, mask, model, model_sigma)

jac_minim = make_jac_func(*args[1:])
_jacf = lambda p, *args: _jac(jac_minim, p, *args)

with tqdm(leave=True) as pbar:
    def update_bar(*args):
            pbar.update(1)
    res = minimize(
        _func, p0.values, args=args,
        jac=_jacf,
        callback=update_bar,
        options=dict(maxiter=100),
    )

final_params = comp_bool_indexing(
    from_1d(pd.Series(res['x'], index=p0.index)).unstack().T
)
"""
from copy import deepcopy
import pandas as pd
from scipy.special import gamma
import jax.numpy as np
from jax import jit, jacfwd
from jax.lax import conv, lgamma
from jax.config import config
from gzbuilder_analysis.config import DEFAULT_DISK
from .sersic import sersic, _b


config.update("jax_enable_x64", True)
EMPTY_COMP = {**DEFAULT_DISK, 'I': 0}


def restructure_model(m):
    m_ = deepcopy(m)
    for k in ('disk', 'bulge', 'bar'):
        if m_[k] is None:
            m_[k] = EMPTY_COMP
    params = [
        params for points, params in m_['spiral']
    ]
    points = [
        np.array(points) for points, params in m_['spiral']
    ]
    m_['spiral'] = dict(
        points=points,
        I=np.array([p['I'] for p in params]),
        spread=np.array([p['spread'] for p in params]),
        falloff=np.array([p['falloff'] for p in params]),
    )
    return m_


@jit
def sersic_ltot(comp):
    return (
        2 * np.pi * comp['I'] * comp['Re']**2 * comp['n']
        * np.exp(_b(comp['n'])) / _b(comp['n'])**(2 * comp['n'])
        * np.exp(lgamma(2 * comp['n']))
    )

@jit
def sersic_I(L, Re, n):
    return L / (
        2 * np.pi * Re**2 * n
        * np.exp(_b(n)) / _b(n)**(2 * n)
        * np.exp(lgamma(2 * n))
    )


# @jit
# def sersic_I(comp):
#     return comp['L'] / (
#         2 * np.pi * comp['Re']**2 * comp['n']
#         * np.exp(_b(comp['n'])) / _b(comp['n'])**(2 * comp['n'])
#         * np.exp(lgamma(2 * comp['n']))
#     )


@jit
def get_fraction(comp_l, disk_l):
    return comp_l / (disk_l + comp_l)


@jit
def get_comp_I(comp, disk):
    comp_re = comp['scale'] * disk['Re']
    comp_l = disk['L'] * comp['frac'] / (1 - comp['frac'])
    return sersic_I(comp_l, comp_re, comp['n'])
    return comp_l / (
        2 * np.pi * comp_re**2 * comp['n']
        * np.exp(_b(comp['n'])) / _b(comp['n'])**(2 * comp['n'])
        * np.exp(lgamma(2 * comp['n']))
    )


@jit
def to_reparametrization(model):
    disk_l = sersic_ltot(model['disk'])
    return dict(
        disk=dict(
            mux=model['disk']['mux'],
            muy=model['disk']['muy'],
            Re=model['disk']['Re'],
            q=model['disk']['q'],
            roll=model['disk']['roll'],
            L=disk_l,
            n=1.0,
            c=2.0,
        ),
        bulge=dict(
            mux=model['bulge']['mux'],
            muy=model['bulge']['muy'],
            scale=model['bulge']['Re'] / model['disk']['Re'],
            q=model['bulge']['q'],
            roll=model['bulge']['roll'],
            frac=get_fraction(sersic_ltot(model['bulge']), disk_l),
            n=model['bulge']['n'],
            c=2.0,
        ),
        bar=dict(
            mux=model['bar']['mux'],
            muy=model['bar']['muy'],
            scale=model['bar']['Re'] / model['disk']['Re'],
            q=model['bar']['q'],
            roll=model['bar']['roll'],
            frac=get_fraction(sersic_ltot(model['bar']), disk_l),
            n=model['bar']['n'],
            c=model['bar']['c'],
        ),
        spiral=model['spiral']
    )


@jit
def from_reparametrization(model):
    return dict(
        disk=dict(
            mux=model['disk']['mux'],
            muy=model['disk']['muy'],
            Re=model['disk']['Re'],
            q=model['disk']['q'],
            roll=model['disk']['roll'],
            I=sersic_I(model['disk']['L'], model['disk']['Re'], 1.0),
            n=1.0,
            c=2.0,
        ),
        bulge=dict(
            mux=model['bulge']['mux'],
            muy=model['bulge']['muy'],
            Re=model['bulge']['scale'] * model['disk']['Re'],
            q=model['bulge']['q'],
            roll=model['bulge']['roll'],
            I=get_comp_I(model['bulge'], model['disk']),
            n=model['bulge']['n'],
            c=2.0,
        ),
        bar=dict(
            mux=model['bar']['mux'],
            muy=model['bar']['muy'],
            Re=model['bar']['scale'] * model['disk']['Re'],
            q=model['bar']['q'],
            roll=model['bar']['roll'],
            I=get_comp_I(model['bar'], model['disk']),
            n=model['bar']['n'],
            c=model['bar']['c'],
        ),
        spiral=model['spiral']
    )


# image manipulation
def asinh(arr):
    """Inverse hyperbolic sine function
    """
    return np.log(arr + np.sqrt(1.0 + (arr * arr)))


def asinh_stretch(px, a=0.6):
    """same as astropy.visualization.AsinhStretch, but with less faff as we
    don't care about being able to invert the transformation.
    """
    return asinh(px / a) / asinh(a)


def downsample(arr, n=5):
    """downsample an array of (n*x, m*y, m) into (x, y, m) using the mean
    """
    shape = (np.asarray(arr.shape) / n).astype(int)
    return arr.reshape(shape[0], n, shape[1], n, -1).mean(3).mean(1)


@jit
def psf_conv(arr, psf):
    """Convolve two 2D arrays
    """
    return conv(
        arr.reshape(1, 1, *arr.shape),
        psf.reshape(1, 1, *psf.shape),
        (1, 1),
        'SAME'
    )[0, 0]


@jit
def norm_nnlf(x):
    """Calclate Negative log-likelihood function for standard normally
    distributed variables `x`
    """
    n = len(x)
    return (
        n / 2 * np.log(2*np.pi)
        # - n / 2 * np.log(1)
        - 1 / 2 * np.nansum(x**2)
    )


@jit
def render(x, y, params, distances, psf):
    disk_p = params['disk']
    bulge_p = params['bulge']
    bar_p = params['bar']
    spiral_p = params['spiral']
    # convert to parametrization needed for rendering
    bulge_re = bulge_p['scale'] * disk_p['Re']
    bar_re = bar_p['scale'] * disk_p['Re']

    disk_I = sersic_I(disk_p['L'], disk_p['Re'], 1.0)
    bulge_I = get_comp_I(bulge_p, disk_p)
    bar_I = get_comp_I(bar_p, disk_p)
    bulge_re = bulge_p['scale'] * disk_p['Re']
    bar_re = bar_p['scale'] * disk_p['Re']

    disk_super = sersic(
        x, y, disk_p['mux'], disk_p['muy'],
        disk_p['roll'], disk_p['q'], 2.0,
        disk_I, disk_p['Re'], 1.0
    )
    bulge_super = sersic(
        x, y, bulge_p['mux'], bulge_p['muy'],
        bulge_p['roll'], bulge_p['q'], 2.0,
        bulge_I, bulge_re, bulge_p['n']
    )
    bar_super = sersic(
        x, y, bar_p['mux'], bar_p['muy'],
        bar_p['roll'], bar_p['q'], 2.0,
        bar_I, bar_re, bar_p['n']
    )
    spiral_disks = downsample(sersic(
        np.expand_dims(x, -1),
        np.expand_dims(y, -1),
        disk_p['mux'], disk_p['muy'],
        disk_p['roll'], disk_p['q'], 2.0,
        disk_I, disk_p['Re'] * spiral_p['falloff'], 1.0
    ))
    spiral = np.sum(
        spiral_p['I']
        * np.exp(-distances**2 / (10 * spiral_p['spread']))
        * spiral_disks,
        axis=-1
    )
    galaxy = downsample(disk_super + bulge_super + bar_super)[:, :, 0] + spiral
    blurred = psf_conv(galaxy, psf)
    return blurred


@jit
def loss(render, target, sigma, mask, params, mu_params, sigma_params):
    """Galaxy Builder loss function (gaussian log-likelihood penalized by
    deviation from aggregated parameters)
    """
    render_delta = (render - target) / sigma
    param_delta = np.concatenate([
        np.reshape(params[k][j] - mu_params[k][j], -1) / sigma_params[k][j]
        for k in params for j in params[k]
    ])
    return -norm_nnlf(np.concatenate((render_delta.ravel(), param_delta)))


@jit
def do_calc(x, y, params, distances, psf, target, sigma, mask,
            mu_params, sigma_params):
    """Render the model, compare it to the data and
    return the negative log-likelihood (with parameter priors)
    """
    r = render(x, y, params, distances, psf)
    nnlf = loss(r, target, sigma, mask, params, mu_params, sigma_params)
    return nnlf


def make_jac_func(x, y, distances, psf, target, sigma, mask,
                  mu_params, sigma_params):
    """Wrapper function to create a jacobian-calculating function accepted by
    scipy.optimize.minimize
    """
    return jacfwd(
        lambda params: do_calc(
            x, y, params, distances, psf,
            target, sigma, mask, mu_params, sigma_params
        )
    )


def comp_bool_indexing(df1):
    """Quickly convert a DataFrame to a dictionary, removing any NaNs
    """
    return {k: v[v.notna()].to_dict() for k, v in df1.items()}


def _func(p, p0, x, y, distances, psf, target, sigma, mask, mu_params, sigma_params):
    """Function to opmtimize in scipy.optimize.minimize
    """
    params = comp_bool_indexing(
        from_1d(
            pd.Series(p, index=p0.index)
        ).unstack().T
    )
    return do_calc(x, y, params, distances, psf, target, sigma, mask,
                   mu_params, sigma_params)


def _jac(jac_minim, p, p0, *args):
    """Function to calculate jacobian for scipy.optimize.minimize
    """
    params = comp_bool_indexing(
        from_1d(
            pd.Series(p, index=p0.index)
        ).unstack().T
    )
    return to_1d(pd.DataFrame(jac_minim(params)).unstack().dropna()).values


def to_1d(params):
    """Unstack a model representation (move spirals.I = [I0, ..., In] to
    spirals.I0, ..., spirals.In)"""
    spiral = params.xs('spiral', drop_level=False)
    no_spiral_params = params.drop(spiral.index)
    new_spirals = spiral.apply(pd.Series).stack()
    new_spirals.index = pd.MultiIndex.from_tuples(
        [(i, f'{j}.{k}') for i, j, k in new_spirals.index]
    )
    return no_spiral_params.combine_first(new_spirals).astype(np.float64)


def from_1d(params):
    """Re-stack a model representation (move spirals.I0, ..., spirals.In to
    spirals.I = [I0, ..., In])"""
    spiral = params.xs('spiral', drop_level=False)
    no_spiral_params = params.drop(spiral.index)
    names = [j.split('.')[0] for i, j in spiral.index.values.reshape(-1, 3)[:, 0]]
    new_spirals = pd.Series(
        (i.astype(np.float64) for i in spiral.values.reshape(-1, 3)),
        index=pd.MultiIndex.from_product([['spiral'], names])
    )
    return no_spiral_params.combine_first(new_spirals)


def get_limits(params_1d):
    """Create a DataFrame of limits to use for fitting, from a reparametrized
    model that has been flattened using `to_1d`
    """
    lims_df = pd.DataFrame([], index=params_1d.index, columns=('lower', 'upper'))

    lims_df['lower'] = -np.inf
    lims_df['upper'] = np.inf

    lims_df.loc[('disk', 'L')] = (0, np.inf)
    lims_df.loc[('disk', 'Re')] = (0.01, np.inf)
    lims_df.loc[('disk', 'q')] = (0.2, 1.2)
    lims_df.drop(('disk', 'n'), inplace=True)  # do not fit disk n
    lims_df.drop(('disk', 'c'), inplace=True)  # do not fit disk c
    if 'bulge' in params_1d.index:
        lims_df.loc[('bulge', 'frac')] = (0, 1-0.01)
        lims_df.loc[('bulge', 'scale')] = (0, 1)
        lims_df.loc[('bulge', 'q')] = (0.6, 1.1)
        lims_df.loc[('bulge', 'n')] = (0.5, 5)
        lims_df.drop(('bulge', 'c'), inplace=True)  # do not fit bulge c
    if 'bar' in params_1d.index:
        lims_df.loc[('bar', 'frac')] = (0, 1-0.01)
        lims_df.loc[('bar', 'scale')] = (0, 1)
        lims_df.loc[('bar', 'q')] = (0.05, 1.2)
        lims_df.loc[('bar', 'n')] = (0.3, 5)
        lims_df.loc[('bar', 'c')] = (0.5, 6)
    i = 0
    while ('spiral', f'I.{i}') in params_1d.index:
        lims_df.loc[('spiral', f'I.{i}')] = (0, np.inf)
        lims_df.loc[('spiral', f'spread.{i}')] = (0, np.inf)
        lims_df.loc[('spiral', f'falloff.{i}')] = (0.01, np.inf)
        i += 1
    # if len(model_obj['spiral']) > 0:
    #     for i in range(len(model_obj['spiral'])):
    #         lims_df.loc[(f'spiral{i}', 'I')] = (0, np.inf)
    #         lims_df.loc[(f'spiral{i}', 'spread')] = (0, np.inf)
    #         lims_df.loc[(f'spiral{i}', 'falloff')] = (0.01, np.inf)

    lims_df.sort_index(inplace=True)
    return lims_df
