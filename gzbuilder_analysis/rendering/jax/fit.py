import pandas as pd
import jax.numpy as np
from jax import jit, ops
from jax.lax import conv
from functools import reduce
from .sersic import sersic, sersic_ltot, sersic_I
from .spiral import vmap_polyline_distance


def comp_bool_indexing(df):
    """Quickly convert a DataFrame to a dictionary, removing any NaNs
    """
    return {k: v[v.notna()].to_dict() for k, v in df.items()}


def _to_dict(values, keys):
    d = {}
    for (k0, k1), v in zip(keys, values):
        d.setdefault(k0, {})
        d[k0][k1] = v
    return d


to_dict = jit(_to_dict, static_argnums=(1,))


@jit
def get_fraction(comp_l, disk_l):
    return comp_l / (disk_l + comp_l)


def to_reparametrization(agg_res, output_pandas=False):
    """Accept an aggregation result and reparametrize the result
    """
    disk = agg_res.params['disk']
    bulge = agg_res.params['bulge']
    bar = agg_res.params['bar']
    spirals = pd.DataFrame([
        agg_res.params[f'spiral{i}']
        for i in range(len(agg_res.spiral_arms))
    ])

    model = pd.DataFrame(
        [],
        columns=['disk', 'bulge', 'bar', 'spiral'],
        dtype=np.float64
    )
    model['disk'] = disk.copy()
    model.loc['L', 'disk'] = sersic_ltot(disk.I, disk.Re, 1.0)
    model.loc['I', 'disk'] = np.nan
    model.loc['n', 'disk'] = np.nan
    model.loc['c', 'disk'] = np.nan

    model['bulge'] = bulge.copy()
    model.loc['scale', 'bulge'] = bulge.Re / disk.Re
    bulge_l = sersic_ltot(bulge.I, bulge.Re, bulge.n)
    model.loc['frac', 'bulge'] = get_fraction(bulge_l, model['disk']['L'])
    model.loc['I', 'bulge'] = np.nan
    model.loc['Re', 'bulge'] = np.nan
    model.loc['c', 'bulge'] = np.nan

    model['bar'] = bar.copy()
    model.loc['scale', 'bar'] = bar.Re / disk.Re
    bar_l = sersic_ltot(bar.I, bar.Re, bar.n)
    model.loc['frac', 'bar'] = get_fraction(bar_l, model['disk']['L'])
    model.loc['I', 'bar'] = np.nan
    model.loc['Re', 'bar'] = np.nan

    for i in range(len(spirals)):
        arm = agg_res.spiral_arms[i]
        model.loc[f'I.{i}', 'spiral'] = spirals['I'].iloc[i]
        model.loc[f'falloff.{i}', 'spiral'] = spirals['falloff'].iloc[i]
        model.loc[f'spread.{i}', 'spiral'] = spirals['spread'].iloc[i]
        model.loc[f'A.{i}', 'spiral'] = arm.A
        model.loc[f'phi.{i}', 'spiral'] = arm.pa * arm.chirality
        model.loc[f't_min.{i}', 'spiral'] = arm.t_predict.min()
        model.loc[f't_max.{i}', 'spiral'] = arm.t_predict.max()
    if output_pandas:
        return model
    return comp_bool_indexing(model)


def get_reparametrized_erros(agg_res):
    disk = agg_res.params['disk']
    bulge = agg_res.params['bulge']
    bar = agg_res.params['bar']

    disk_e = agg_res.errors['disk']
    bulge_e = agg_res.errors['bulge']
    bar_e = agg_res.errors['bar']

    errs = pd.DataFrame(
        [],
        columns=['disk', 'bulge', 'bar', 'spiral'],
        dtype=np.float64
    )
    errs['disk'] = disk_e.copy()
    errs.loc['L', 'disk'] = np.inf
    errs.loc['I', 'disk'] = np.nan
    errs.loc['n', 'disk'] = np.nan
    errs.loc['c', 'disk'] = np.nan

    errs['bulge'] = bulge_e.copy()
    errs.loc['scale', 'bulge'] = bulge.Re / disk.Re * np.sqrt(
        bulge_e.Re**2 / bulge.Re**2
        + disk_e.Re**2 / disk.Re**2
    )
    errs.loc['frac', 'bulge'] = np.inf
    errs.loc['I', 'bulge'] = np.nan
    errs.loc['Re', 'bulge'] = np.nan
    errs.loc['c', 'bulge'] = np.nan

    errs['bar'] = bar_e.copy()
    errs.loc['scale', 'bar'] = bar.Re / disk.Re * np.sqrt(
        bar_e.Re**2 / bar.Re**2
        + disk_e.Re**2 / disk.Re**2
    )
    errs.loc['frac', 'bar'] = np.inf
    errs.loc['I', 'bar'] = np.nan
    errs.loc['Re', 'bar'] = np.nan

    for i in range(len(agg_res.spiral_arms)):
        errs.loc[f'I.{i}', 'spiral'] = np.inf
        errs.loc[f'falloff.{i}', 'spiral'] = np.inf
        errs.loc[f'spread.{i}', 'spiral'] = np.inf
        errs.loc[f'A.{i}', 'spiral'] = 0.01
        errs.loc[f'phi.{i}', 'spiral'] = 1
        errs.loc[f't_min.{i}', 'spiral'] = np.deg2rad(0.5)
        errs.loc[f't_max.{i}', 'spiral'] = np.deg2rad(0.5)
    return comp_bool_indexing(errs)


def get_limits(agg_res):
    n_spirals = len(agg_res.spiral_arms)
    return {
        'disk': {
            'L': [0.0, np.inf],
            'mux': [-np.inf, np.inf],
            'muy': [-np.inf, np.inf],
            'q': [0.3, 1.2],
            'roll': [-np.inf, np.inf],
            'Re': [0.01, np.inf],
        },
        'bulge': {
            'c': [],
            'frac': [0.0, 0.99],
            'mux': [-np.inf, np.inf],
            'muy': [-np.inf, np.inf],
            'n': [0.5, 5],
            'q': [0.6, 1.2],
            'roll': [-np.inf, np.inf],
            'scale': [0.05, 1],
        },
        'bar': {
            'c': [1, 6],
            'frac': [0.0, 0.99],
            'mux': [-np.inf, np.inf],
            'muy': [-np.inf, np.inf],
            'n': [0.3, 5],
            'q': [0.05, 0.6],
            'roll': [-np.inf, np.inf],
            'scale': [0.05, 1],
        },
        'spiral': reduce(lambda a, b: {**a, **b}, ({
            f'I.{i}': [0, np.inf],
            f'A.{i}': [0, np.inf],
            f'falloff.{i}': [0.001, np.inf],
            f'phi.{i}': [-85.0, 85.0],
            f'spread.{i}': [0.05, np.inf],
            f't_min.{i}': [-np.inf, np.inf],
            f't_max.{i}': [-np.inf, np.inf],
        } for i in range(n_spirals))) if n_spirals > 0 else {}
    }


def _logsp(t_min, t_max, A, phi, q, roll, mux, muy, N):
    theta = np.linspace(t_min, t_max, N)
    Rls = A * np.exp(np.tan(np.deg2rad(phi)) * theta)
    rot_matrix = np.array((
        (np.cos(-roll), np.sin(-roll)),
        (-np.sin(-roll), np.cos(-roll))
    ))
    return np.dot(
        rot_matrix,
        Rls * np.array((q * np.cos(theta), np.sin(theta)))
    ).T + np.array((mux, muy))


logsp = jit(_logsp, static_argnums=(8,))


def log_spiral(t_min=0, t_max=2*np.pi, A=0.1, phi=10, q=0, roll=0, mux=0, muy=0,
               N=200, **kwargs):
    return logsp(t_min, t_max, A, phi, q, roll, mux, muy, N)


def downsample(arr, n=5):
    """downsample an array of (n*x, m*y, m) into (x, y, m) using the mean
    """
    shape = (np.asarray(arr.shape) / n).astype(int)
    return arr.reshape(shape[0], n, shape[1], n, -1).mean(3).mean(1)


@jit
def psf_conv(arr, psf):
    return conv(
        arr.reshape(1, 1, *arr.shape),
        psf.reshape(1, 1, *psf.shape),
        (1, 1),
        'SAME'
    )[0, 0]


def _render(x, y, params, distances, psf, n_spirals):
    disk_p = params['disk']
    bulge_p = params['bulge']
    bar_p = params['bar']
    spiral_p = params['spiral']
    # convert to parametrization needed for rendering
    bulge_re = bulge_p['scale'] * disk_p['Re']
    bar_re = bar_p['scale'] * disk_p['Re']

    disk_I = sersic_I(disk_p['L'], disk_p['Re'], 1.0)
    bulge_l = disk_p['L'] * bulge_p['frac'] / (1 - bulge_p['frac'])
    bulge_I = sersic_I(bulge_l, bulge_re, bulge_p['n'])
    bar_l = disk_p['L'] * bar_p['frac'] / (1 - bar_p['frac'])
    bar_I = sersic_I(bar_l, bar_re, bar_p['n'])

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
        bar_p['roll'], bar_p['q'], bar_p['c'],
        bar_I, bar_re, bar_p['n']
    )

    Is = np.array([spiral_p[f'I.{i}'] for i in range(n_spirals)])
    spreads = np.array([spiral_p[f'spread.{i}'] for i in range(n_spirals)])
    falloffs = np.array([spiral_p[f'falloff.{i}'] for i in range(n_spirals)])
    spiral_disks = downsample(sersic(
        np.expand_dims(x, -1),
        np.expand_dims(y, -1),
        disk_p['mux'], disk_p['muy'],
        disk_p['roll'], disk_p['q'], 2.0,
        disk_I, disk_p['Re'] * falloffs, 1.0
    ))
    spiral = np.sum(
        Is
        * np.exp(-distances**2 / (10 * spreads))
        * spiral_disks,
        axis=-1
    )
    galaxy = downsample(disk_super + bulge_super + bar_super)[:, :, 0] + spiral
    blurred = psf_conv(galaxy, psf)
    return blurred


render = jit(_render, static_argnums=(0, 1, 4, 5))


@jit
def norm_nnlf(x):
    """Calclate Negative log-likelihood function for standard normally
    distributed variables `x`
    """
    n = len(x)
    return -(
        n / 2 * np.log(2*np.pi)
        # - n / 2 * np.log(1)
        - 1 / 2 * np.nansum(x**2)
    )


def _make_xy_arrays(target, On):
    x = np.arange(target.shape[1], dtype=np.float64)
    y = np.arange(target.shape[0], dtype=np.float64)
    cx, cy = np.meshgrid(x, y)
    x_super = np.linspace(0.5 / On - 0.5, target.shape[1] - 0.5 - 0.5 / On,
                          target.shape[1] * On)
    y_super = np.linspace(0.5 / On - 0.5, target.shape[0] - 0.5 - 0.5 / On,
                          target.shape[0] * On)
    cx_super, cy_super = np.meshgrid(x_super, y_super)
    return (cx, cy), (cx_super, cy_super)


def _get_distances(cx, cy, model, n_spirals):
    if n_spirals > 0:
        spirals = [
            # t_min, t_max, A, phi, q, roll, mux, muy, N
            logsp(
                model['spiral'][f't_min.{i}'], model['spiral'][f't_max.{i}'],
                model['spiral'][f'A.{i}'], model['spiral'][f'phi.{i}'],
                model['disk']['q'], model['disk']['roll'],
                model['disk']['mux'], model['disk']['muy'],
                200,
            )
            for i in range(n_spirals)
        ]
        distances = np.stack([
            vmap_polyline_distance(s, cx, cy)
            for s in spirals
        ], axis=-1)
    else:
        distances = np.array([], dtype=np.float64)
    return distances


def _step(p, keys, n_spirals, base_model, model_err, psf, mask, target, sigma,
          On=5):
    (cx, cy), (cx_super, cy_super) = _make_xy_arrays(target, On)

    # (1/6) get model dict from vector
    new_params = to_dict(p, keys)
    model = {k: {**base_model[k], **new_params.get(k, {})} for k in base_model}

    # (2/6) obtain spiral arms from parameters and calculate distance matrices
    distances = _get_distances(cx, cy, model, n_spirals)

    # (3/6) render the model
    r = render(cx_super, cy_super, model, distances, psf, n_spirals)

    # (4/6) calculate the model's NLL
    render_delta = (r - target) / sigma
    masked_render_delta = ops.index_update(render_delta, mask, np.nan)
    model_nll = np.nansum(norm_nnlf(masked_render_delta.ravel()))

    # (5/6) calculate the parameter NLL (deviation from initial conditions)
    mu_p = np.array([base_model[k0][k1] for k0, k1 in keys])
    sigma_p = np.array([model_err[k0][k1] for k0, k1 in keys])
    param_delta = (p - mu_p) / sigma_p
    masked_param_delta = ops.index_update(
        param_delta, (sigma_p == np.inf), np.nan
    )
    param_nll = np.nansum(norm_nnlf(masked_param_delta.ravel()))

    # (6/6) return the sum of the NLLs
    return model_nll + param_nll


step = jit(_step, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8))


# recreates the first steps in the `step`, for reproducability
def _create_model(p, keys, n_spirals, base_model, psf, target, On=5):
    (cx, cy), (cx_super, cy_super) = _make_xy_arrays(target, On)
    # (1/6) get model dict from vector
    new_params = to_dict(p, keys)
    model = {k: {**base_model[k], **new_params.get(k, {})} for k in base_model}
    # (2/6) obtain spiral arms from parameters and calculate distance matrices
    distances = _get_distances(cx, cy, model, n_spirals)
    # (3/6) render the model
    return render(cx_super, cy_super, model, distances, psf, n_spirals)


create_model = jit(_create_model, static_argnums=(1, 2, 3, 4, 5, 6))
