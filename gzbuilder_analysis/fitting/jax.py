import pandas as pd
import jax.numpy as np
from jax import jit, jacrev
from jax.lax import conv
from functools import reduce
from ..rendering.jax.sersic import sersic, sersic_ltot, sersic_I
from ..rendering.jax.spiral import vmap_polyline_distance
from gzbuilder_analysis.config import COMPONENT_PARAM_BOUNDS


EMPTY_SERSIC = pd.Series(
    dict(mux=np.nan, muy=np.nan, Re=0.5, roll=0, q=1, I=0, n=1, c=2)
)
EMPTY_SERSIC_ERR = pd.Series(
    dict(mux=np.nan, muy=np.nan, Re=0.1, roll=0.01, q=0.01, I=0.01, n=0.01, c=0.01)
)


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
    if 'bulge' in agg_res.params:
        bulge = agg_res.params['bulge']
    else:
        bulge = EMPTY_SERSIC
    if 'bar' in agg_res.params:
        bar = agg_res.params['bar']
    else:
        bar = EMPTY_SERSIC

    model = pd.DataFrame(
        [],
        columns=['disk', 'bulge', 'bar', 'spiral', 'centre'],
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

    model.loc['mux', 'centre'] = np.nanmean(
        np.array((model.bulge.mux, model.bar.mux))
    )
    model.loc['muy', 'centre'] = np.nanmean(
        np.array((model.bulge.muy, model.bar.muy))
    )
    if np.isnan(model.loc['mux', 'centre']):
        model.loc['mux', 'centre'] = 0
    if np.isnan(model.loc['muy', 'centre']):
        model.loc['muy', 'centre'] = 0
    model.loc['mux', 'bulge'] = np.nan
    model.loc['muy', 'bulge'] = np.nan
    model.loc['mux', 'bar'] = np.nan
    model.loc['muy', 'bar'] = np.nan

    model = pd.concat((
        model, agg_res.params.get('spiral', pd.Series({})).rename('spiral')
    ), sort=False, axis=1)
    if output_pandas:
        return model.unstack().dropna()
    return comp_bool_indexing(model)


def get_reparametrized_erros(agg_res):
    disk = agg_res.params['disk']
    if 'bulge' in agg_res.params:
        bulge = agg_res.params['bulge']
    else:
        bulge = EMPTY_SERSIC
    if 'bar' in agg_res.params:
        bar = agg_res.params['bar']
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
    return comp_bool_indexing(errs)


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


# this is a duplicate of "inclined_log_spiral" from aggregation.spirals.__init__
# using jax
@jit
def rot_matrix(a):
    return np.array((
        (np.cos(a), np.sin(a)),
        (-np.sin(a), np.cos(a))
    ))


def _logsp(t_min, t_max, A, phi, q, psi, dpsi, mux, muy, N):
    theta = np.linspace(t_min, t_max, N)
    Rls = A * np.exp(np.tan(np.deg2rad(phi)) * theta)
    qmx = np.array(((q, 0), (0, 1)))
    mx = np.dot(rot_matrix(-psi), np.dot(qmx, rot_matrix(dpsi)))
    return (
        Rls * np.dot(mx, np.array((np.cos(theta), np.sin(theta))))
    ).T + np.array([mux, muy])


logsp = jit(_logsp, static_argnums=(9,))


def log_spiral(t_min=0, t_max=2*np.pi, A=0.1, phi=10, q=0, roll=0,
               delta_roll=0, mux=0, muy=0, N=200, **kwargs):
    return logsp(t_min, t_max, A, phi, q, roll, delta_roll, mux, muy, N)


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


def from_reparametrization(params):
    # requires a disk, bulge and bar, which can have a fraction / scale of 0
    disk_p = params.get('disk')
    bulge_p = params.get('bulge')
    bar_p = params.get('bar')
    centre_pos = params.get('centre')
    spiral_p = params.get('spiral', {})

    disk_I = sersic_I(disk_p['L'], disk_p['Re'], 1.0)
    bulge_re = bulge_p['scale'] * disk_p['Re']
    bulge_l = disk_p['L'] * bulge_p['frac'] / (1 - bulge_p['frac'])
    bulge_I = sersic_I(bulge_l, bulge_re, bulge_p['n'])
    bar_re = bar_p['scale'] * disk_p['Re']
    bar_l = disk_p['L'] * bar_p['frac'] / (1 - bar_p['frac'])
    bar_I = sersic_I(bar_l, bar_re, bar_p['n'])

    return dict(
        disk=dict(
            mux=disk_p['mux'], muy=disk_p['muy'],
            q=disk_p['q'], Re=disk_p['Re'], roll=disk_p['roll'],
            I=disk_I,
        ),
        bulge=dict(
            mux=centre_pos['mux'], muy=centre_pos['muy'],
            q=bulge_p['q'], roll=bulge_p['roll'], n=bulge_p['n'],
            Re=bulge_re, I=bulge_I,
        ),
        bar=dict(
            mux=centre_pos['mux'], muy=centre_pos['muy'],
            q=bar_p['q'], roll=bar_p['roll'], n=bar_p['n'], c=bar_p['c'],
            Re=bar_re, I=bar_I,
        ),
        spiral=spiral_p,
    )


def remove_invisible_components(model):
    for k in ('disk', 'bulge', 'bar'):
        if model[k]['I'] == 0.0 or model[k]['Re'] == 0.0:
            model[k] = None
    return model


def _render(x, y, params, distances, psf, n_spirals):
    model = from_reparametrization(params)
    disk_I = sersic_I(params['disk']['L'], params['disk']['Re'], 1.0)
    bulge_re = params['bulge']['scale'] * params['disk']['Re']
    bulge_l = params['disk']['L'] * params['bulge']['frac'] / (1 - params['bulge']['frac'])
    bulge_I = sersic_I(bulge_l, bulge_re, params['bulge']['n'])

    disk_super = sersic(
        x, y, model['disk']['mux'], model['disk']['muy'],
        model['disk']['roll'], model['disk']['q'], 2.0,
        model['disk']['I'], model['disk']['Re'], 1.0
    )
    bulge_super = sersic(
        x, y, model['bulge']['mux'], model['bulge']['muy'],
        model['bulge']['roll'], model['bulge']['q'], 2.0,
        model['bulge']['I'], model['bulge']['Re'], model['bulge']['n']
    )
    bar_super = sersic(
        x, y, model['bar']['mux'], model['bar']['muy'],
        model['bar']['roll'], model['bar']['q'], model['bar']['c'],
        model['bar']['I'], model['bar']['Re'], model['bar']['n']
    )

    Is = np.array([model['spiral']['I.{}'.format(i)] for i in range(n_spirals)])
    spreads = np.array([
        model['spiral']['spread.{}'.format(i)] for i in range(n_spirals)
    ])
    spiral_disks = downsample(sersic(
        np.expand_dims(x, -1),
        np.expand_dims(y, -1),
        model['disk']['mux'], model['disk']['muy'],
        model['disk']['roll'], model['disk']['q'], 2.0,
        model['disk']['I'], model['disk']['Re'], 1.0
    ))
    spiral = np.sum(
        Is
        * np.exp(-distances**2 / (2*spreads**2))
        * spiral_disks,
        axis=-1
    )
    galaxy = downsample(disk_super + bulge_super + bar_super)[:, :, 0] + spiral
    blurred = psf_conv(galaxy, psf)
    return blurred


render = jit(_render, static_argnums=(0, 1, 4, 5))


@jit
def norm_nnlf(x):
    r"""Calclate Negative log-likelihood function for standard normally
    distributed variables `x`
    $$
    l(\mu, \sigma; x_1, ..., x_n) =
        -\frac{n}{2}\ln(2\pi)
        -\frac{n}{2}\ln(\sigma^2)
        - \frac{1}{2\sigma^2}\sum_{j=1}^n(x_j - \mu)^2
    $$
    """
    n = len(x)
    log_likelihood = (
        -n / 2 * np.log(2*np.pi)
        - 1 / 2 * np.sum(x**2)
    )
    return -log_likelihood


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


def _get_distances(cx, cy, model, base_model, n_spirals):
    delta_roll = model['disk']['roll'] - base_model['disk']['roll']
    if n_spirals > 0:
        spirals = [
            # t_min, t_max, A, phi, q, psi, dpsi, mux, muy, N
            logsp(
                model['spiral']['t_min.{}'.format(i)],
                model['spiral']['t_max.{}'.format(i)],
                model['spiral']['A.{}'.format(i)],
                model['spiral']['phi.{}'.format(i)],
                model['disk']['q'],
                model['disk']['roll'],
                delta_roll,
                model['disk']['mux'],
                model['disk']['muy'],
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


def make_step(keys, n_spirals, base_model, model_err, psf, mask, target,
              sigma, On=5):
    (cx, cy), (cx_super, cy_super) = _make_xy_arrays(target, On)

    def step(p):
        # (1/6) get model dict from vector
        new_params = to_dict(p, keys)
        model = {k: {**base_model[k], **new_params.get(k, {})} for k in base_model}
        # (2/6) obtain spiral arms from parameters and calculate distance matrices
        distances = _get_distances(cx, cy, model, base_model, n_spirals)

        # (3/6) render the model
        r = render(cx_super, cy_super, model, distances, psf, n_spirals)

        # (4/6) calculate the model's NLL
        render_delta = (r - target) / sigma
        masked_render_delta = render_delta[~mask]
        model_nll = norm_nnlf(masked_render_delta.ravel())

        # (5/6) calculate the parameter NLL (deviation from initial conditions)
        mu_p = np.array([base_model[k0][k1] for k0, k1 in keys])
        sigma_p = np.array([model_err[k0][k1] for k0, k1 in keys])
        param_delta = (p - mu_p) / sigma_p
        masked_param_delta = param_delta[sigma_p != np.inf]
        param_nll = norm_nnlf(masked_param_delta.ravel())

        # (6/6) return the sum of the NLLs
        return model_nll + param_nll
    return jit(step)


# recreates the first steps in the `step`, for reproducability
def _create_model(p, keys, n_spirals, base_model, psf, target, On=5):
    (cx, cy), (cx_super, cy_super) = _make_xy_arrays(target, On)
    # (1/6) get model dict from vector
    new_params = to_dict(p, keys)
    model = {k: {**base_model[k], **new_params.get(k, {})} for k in base_model}
    # (2/6) obtain spiral arms from parameters and calculate distance matrices
    distances = _get_distances(cx, cy, model, base_model, n_spirals)
    # (3/6) render the model
    return render(cx_super, cy_super, model, distances, psf, n_spirals)


create_model = jit(_create_model, static_argnums=(1, 2, 3, 4, 5, 6))


class Optimizer():
    def __init__(self, aggregation_result, fm, components='all'):
        self.components = components
        self.aggregation_result = aggregation_result
        self.model = to_reparametrization(aggregation_result)
        self.model_err = get_reparametrized_erros(aggregation_result)
        self.n_spirals = (
            len(aggregation_result.spiral_arms)
            if 'spiral' in components or 'all' in components
            else 0
        )
        self.psf = fm.psf
        self.target = np.asarray(fm.galaxy_data.data)
        self.mask = np.asarray(fm.galaxy_data.mask)
        self.sigma = np.asarray(fm.sigma_image.data)
        self.reset_keys()

    def __getitem__(self, keys):
        if type(keys) in {list, tuple}:
            return self.model[keys[0]][keys[1]]
        else:
            return self.model[keys]

    def __setitem__(self, keys, value):
        if type(keys) in {list, tuple}:
            self.model.setdefault(keys[0], {})
            self.model[keys[0]].update({keys[1]: value})
        else:
            self.model[keys] = value
        self.__prep()

    def __prep(self):
        self.p0 = np.array([
            self.model[k0][k1] for k0, k1 in self.keys
        ])
        self.sigma_param = np.array([
            self.model_err[k0][k1] for k0, k1 in self.keys
        ])
        self.lims = get_limits(self.aggregation_result)
        self.limits = [self.lims[k0][k1] for k0, k1 in self.keys]
        self.step = make_step(
            self.keys, self.n_spirals,
            self.model, self.model_err,
            self.psf, self.mask, self.target,
            self.sigma
        )
        self.jacobian = jit(jacrev(self.step))

    def __call__(self, p):
        return self.step(p)

    def set_keys(self, new_keys):
        self.keys = new_keys
        self.__prep()

    def reset_keys(self):
        self.keys = [
            (k0, k1)
            for k0 in self.model
            for k1 in self.model[k0]
            if (
                'falloff' not in k1
                and (
                    k0 in self.aggregation_result.params
                    and (k0 in self.components or self.components == 'all')
                )
                or (
                    'spiral' in k0
                    and ('spiral' in self.components or self.components == 'all')
                )
            )
        ]
        # fix bulge and bar position to some "centre" position
        if any(k0 in {'bulge', 'bar'} for k0, _ in self.keys):
            self.keys.extend([('centre', 'mux'), ('centre', 'muy')])

        self.__prep()

    def get_spirals(self):
        delta_roll = (
            self.model['disk']['roll']
            - self.aggregation_result.params['disk']['roll']
        )
        return np.array([
            logsp(
                self.model['spiral']['t_min.{}'.format(i)],
                self.model['spiral']['t_max.{}'.format(i)],
                self.model['spiral']['A.{}'.format(i)],
                self.model['spiral']['phi.{}'.format(i)],
                self.model['disk']['q'],
                self.model['disk']['roll'],
                delta_roll,
                self.model['disk']['mux'],
                self.model['disk']['muy'],
                100,
            )
            for i in range(self.n_spirals)
        ])

    def render(self, p):
        return create_model(
            p, self.keys, self.n_spirals, self.model, self.psf, self.target, 5
        )
