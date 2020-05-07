import re
import jax.numpy as np
from jax import jit
from jax.lax import conv
import pandas as pd
from gzbuilder_analysis.rendering.spiral import correct_logsp_params
from gzbuilder_analysis import df_to_dict


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


def correct_spirals(model, base_roll):
    model_ = model.copy()
    if 'spiral' not in model:
        return model_
    spirals = model['spiral']
    dpsi = model[('disk', 'roll')] - base_roll
    for i in range(100):
        if 'A.{}'.format(i) not in spirals:
            continue
        # correct_logsp_params(A, phi, q, psi, dpsi, theta)
        A, phi, q, roll, (t_min, t_max) = correct_logsp_params(
            spirals['A.{}'.format(i)],
            spirals['phi.{}'.format(i)],
            model_['disk']['q'],
            model_['disk']['roll'],
            dpsi,
            np.array((
                spirals['t_min.{}'.format(i)],
                spirals['t_max.{}'.format(i)],
            )),
        )
        new_spiral = pd.Series({
            ('spiral', 'A.{}'.format(i)): A,
            ('spiral', 'phi.{}'.format(i)): phi,
            ('spiral', 't_min.{}'.format(i)): t_min,
            ('spiral', 't_max.{}'.format(i)): t_max,
        })
        model_.update(new_spiral)
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
        if model[(comp, 'q')] > 1:
            size_param = 'Re' if comp == 'disk' else 'scale'

            model_.update(pd.Series({
                (comp, 'q'): 1 / model[(comp, 'q')],
                (comp, size_param): (
                    model[(comp, 'q')]
                    * model[(comp, size_param)]
                ),
                (comp, 'roll'): (
                    (model[(comp, 'roll')] + np.pi/2) % np.pi
                ),
            }))
    if model[('disk', 'q')] > 1:
        # if we have altered the disk, make sure to update the scales of the
        # other components
        if 'bulge' in model:
            model_[('bulge', 'scale')] /= model[('disk', 'q')]
        if 'bar' in model:
            model_[('bar', 'scale')] /= model[('disk', 'q')]

        # if we have altered the disk, make sure we correct the spirals
        if 'spiral' in model:
            indices = {int(c.split('.')[1]) for c in model['spiral'].index}
            original_q = model[('disk', 'q')]
            for i in indices:
                A, phi, q, roll, (t_min, t_max) = correct_logsp_params(
                    model_[('spiral', 'A.{}'.format(i))] * original_q,
                    model_[('spiral', 'phi.{}'.format(i))],
                    model_[('disk', 'q')],
                    model_[('disk', 'roll')],
                    -np.pi / 2,
                    np.array((
                        model_[('spiral', 't_min.{}'.format(i))],
                        model_[('spiral', 't_max.{}'.format(i))],
                    ))
                )
                new_spiral = pd.Series({
                    ('spiral', 'A.{}'.format(i)): A,
                    ('spiral', 'phi.{}'.format(i)): phi,
                    ('spiral', 't_min.{}'.format(i)): t_min,
                    ('spiral', 't_max.{}'.format(i)): t_max,
                })
                model_.update(new_spiral)
    return model_


def GZB_score(D):
    """Recreate the score shown to volunteers from an (unscaled) difference
    image
    """
    N = np.multiply.reduce(D.shape)
    return 100 * np.exp(
        -300 / N
        * np.sum(
            asinh(np.abs(D) / 0.6)**2
            / asinh(0.6)
        )
    )
