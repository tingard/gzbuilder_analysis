import re
import jax.numpy as np
from jax import jit
from jax.lax import conv
import pandas as pd
from gzbuilder_analysis.rendering.jax.spiral import correct_logsp_params


def df_to_dict(df):
    """Quickly convert a DataFrame to a dictionary, removing any NaNs
    """
    return {k: v[v.notna()].to_dict() for k, v in df.items()}


def _make_xy_arrays(shape, On):
    x = np.arange(shape[1], dtype=np.float64)
    y = np.arange(shape[0], dtype=np.float64)
    cx, cy = np.meshgrid(x, y)
    x_super = np.linspace(0.5 / On - 0.5, shape[1] - 0.5 - 0.5 / On,
                          shape[1] * On)
    y_super = np.linspace(0.5 / On - 0.5, shape[0] - 0.5 - 0.5 / On,
                          shape[0] * On)
    cx_super, cy_super = np.meshgrid(x_super, y_super)
    return (cx, cy), (cx_super, cy_super)


def downsample(arr, n=5):
    """downsample an array of (n*x, m*y, m) into (x, y, m) using the mean
    """
    shape = (np.asarray(arr.shape) / n).astype(int)
    return arr.reshape(shape[0], n, shape[1], n, -1).mean(3).mean(1)


downsample = jit(downsample, static_argnums=(1,))


@jit
def psf_conv(arr, psf):
    return conv(
        arr.reshape(1, 1, *arr.shape),
        psf.reshape(1, 1, *psf.shape),
        (1, 1),
        'SAME'
    )[0, 0]


def get_luminosity_keys(model):
    L_keys = [('disk', 'L'), ('bulge', 'frac'), ('bar', 'frac')]
    if 'spiral' in model:
        L_keys += [
            i for i in model.xs('spiral', drop_level=False).index
            if re.match(r'I\.[0-9]+', i[1])
        ]
    return [i for i in L_keys if i in model]


def remove_zero_brightness_components(model):
    model_ = model.copy()
    L_keys = get_luminosity_keys(model)
    for k, val in model_.where(model_[L_keys] == 0).dropna().items():
        if k[0] != 'spiral':
            model_.drop(k[0], level=0, inplace=True)
        else:
            idx = int(k[1].split('.')[1])
            model_.drop([
                ('spiral', c)
                for c in model_.xs('spiral').index
                if re.match(r'.*?\.{}'.format(idx), c)
            ], inplace=True)
    return model_


def lower_spiral_indices(model):
    model_ = model.copy()
    if 'spiral' not in model:
        return model_
    i = j = 0
    mx = max(int(i.split('.')[1]) for i in model_.xs('spiral').index)
    for _ in range(mx + 1):
        if ('spiral', 'A.{}'.format(i)) in model_:
            if i != j:
                keys = [
                    ('spiral', '{}.{}'.format(v, i))
                    for v in ('I', 'spread', 'A', 'phi', 't_min', 't_max')
                ]
                new_keys = [
                    ('spiral', '{}.{}'.format(v, j))
                    for v in ('I', 'spread', 'A', 'phi', 't_min', 't_max')
                ]
                model_ = pd.Series(
                    model_[keys].values,
                    pd.MultiIndex.from_tuples(new_keys)
                ).combine_first(model_.drop(keys))
            i += 1
            j += 1
            continue
        else:
            i += 1
    return model_


def correct_axratio(model):
    model_ = model.copy()
    for comp in ('disk', 'bulge', 'bar'):
        if comp not in model:
            continue
        if model_[(comp, 'q')] > 1:
            size_param = 'Re' if comp == 'disk' else 'scale'
            model_.update(pd.Series({
                (comp, 'q'): 1 / model_[(comp, 'q')],
                (comp, size_param): (
                    model_[(comp, 'q')]
                    * model_[(comp, size_param)]
                ),
                (comp, 'roll'): (
                    (model_[(comp, 'roll')] + np.pi/2) % np.pi
                ),
            }))
    return model_


def correct_spirals(model, base_roll):
    model_ = model.copy()
    if 'spiral' not in model:
        return model_
    spirals = model['spiral']
    for i in range(len(spirals) // 6):
        A, phi, q, roll, (t_min, t_max) = correct_logsp_params(
            spirals['A.{}'.format(i)],
            spirals['phi.{}'.format(i)],
            model_['disk']['q'],
            model_['disk']['roll'],
            model_['disk']['roll'] - base_roll,
            np.array((
                spirals['t_min.{}'.format(i)],
                spirals['t_max.{}'.format(i)],
            )),
        )
        model_['spiral'].update(pd.Series({
            'A.{}'.format(i): A,
            'phi.{}'.format(i): phi,
            't_min.{}'.format(i): t_min,
            't_max.{}'.format(i): t_max,
        }))
    return model_
