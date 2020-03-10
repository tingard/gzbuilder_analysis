from copy import deepcopy
import jax.numpy as np
import numpy as onp
from scipy.optimize import minimize
from tqdm import tqdm
from jax import jit, jacrev
from gzbuilder_analysis.rendering.jax.sersic import sersic, sersic_I
from gzbuilder_analysis.rendering.jax.spiral import translate_spiral, \
    corrected_inclined_lsp, vmap_polyline_distance, correct_logsp_params
from .misc import _make_xy_arrays, downsample, psf_conv
from .nnlf import negative_log_likelihood
from .reparametrization import to_reparametrization, \
    get_reparametrized_errors, get_limits


EMPTY_SERSIC = dict(mux=np.nan, muy=np.nan, Re=0.5, roll=0, q=1, I=0, n=1, c=2)


def get_p(d, keys):
    return np.array([d[k0][k1] for k0, k1 in keys])


get_p = jit(get_p, static_argnums=(1,))


@jit
def from_frac(frac, disk_l):
    disk_l * frac / (1 - frac)


def get_spirals(model, n_spirals, base_roll):
    dpsi = model['disk']['roll'] - base_roll
    return [
        translate_spiral(
            corrected_inclined_lsp(
                # (A, phi, q, psi, dpsi, theta)
                model['spiral']['A.{}'.format(i)],
                model['spiral']['phi.{}'.format(i)],
                model['disk']['q'],
                model['disk']['roll'],
                dpsi,
                np.linspace(
                    model['spiral']['t_min.{}'.format(i)],
                    model['spiral']['t_max.{}'.format(i)],
                    100
                ),
            ),
            model['disk']['mux'],
            model['disk']['muy'],
        )
        for i in range(n_spirals)
        if 'A.{}'.format(i) in model['spiral']
    ]


def get_spiral_distances(cx, cy, spirals):
    return np.stack([
        vmap_polyline_distance(s, cx, cy)
        for s in spirals
    ], axis=-1)


def render_spirals(cx, cy, model, disk_I, n_spirals, distances, disk):
    Is = np.array([
        model['spiral']['I.{}'.format(i)]
        for i in range(n_spirals)
    ])
    spreads = np.array([
        model['spiral']['spread.{}'.format(i)] for i in range(n_spirals)
    ])
    return np.sum(
        Is
        * np.exp(-distances**2 / (2*spreads**2))
        * np.expand_dims(disk, -1),
        axis=-1
    )


def _make_render_func(model, shape, oversample_n):
    """Accepts a reparametrized model, an output_shape, an oversample factor and
    and renders galaxy builder components
    """
    P, P_super = _make_xy_arrays(shape, oversample_n)
    n_spirals = len(model.get('spiral', [])) // 6
    base_roll = model['disk']['roll']
    comps = {
        k: bool(model.get(k, False))
        for k in ('disk', 'bulge', 'bar', 'spiral')
    }

    def render(new_model, has_bulge, has_bar, n_spirals):
        out = {}
        m_disk = new_model['disk']
        disk_I = sersic_I(m_disk['L'], m_disk['Re'], m_disk['q'], 1, 2)
        # mux=0, muy=0, roll=0, q=1, c=2, I=1, Re=1, n=1
        disk_super = sersic(
            *P_super,
            mux=m_disk['mux'], muy=m_disk['muy'],
            roll=m_disk['roll'], q=m_disk['q'], Re=m_disk['Re'], I=disk_I,
            n=1.0, c=2.0,
        )
        out['disk'] = np.squeeze(downsample(disk_super, oversample_n))
        if n_spirals > 0:
            spirals = get_spirals(new_model, n_spirals, base_roll)
            spiral_distances = get_spiral_distances(*P, spirals)
            spirals = render_spirals(
                *P, new_model, disk_I, n_spirals, spiral_distances, out['disk']
            )
            out['spiral'] = spirals
        else:
            spirals = np.zeros(shape)
        disk_spiral_L = out['disk'].sum() + spirals.sum()

        if has_bulge:
            m_bulge = new_model['bulge']
            bulge_L = (
                m_bulge['frac'] * (disk_spiral_L)
                / (1 - m_bulge['frac'])
            )
            bulge_Re = m_disk['Re'] * m_bulge['scale']
            bulge_I = sersic_I(
                bulge_L, bulge_Re, m_bulge['q'], m_bulge['n']
            )
            bulge_super = sersic(
                *P_super,
                mux=new_model['centre']['mux'], muy=new_model['centre']['muy'],
                roll=m_bulge['roll'], q=m_bulge['q'], Re=bulge_Re, I=bulge_I,
                n=m_bulge['n'], c=2.0
            )
            out['bulge'] = np.squeeze(downsample(bulge_super, oversample_n))
        if has_bar:
            m_bar = new_model['bar']
            bar_L = (
                m_bar['frac'] * (disk_spiral_L)
                / (1 - m_bar['frac'])
            )
            bar_Re = m_disk['Re'] * m_bar['scale']
            bar_I = sersic_I(bar_L, bar_Re, m_bar['q'], m_bar['n'])
            bar_super = sersic(
                *P_super,
                mux=new_model['centre']['mux'], muy=new_model['centre']['muy'],
                roll=m_bar['roll'], q=m_bar['q'], Re=bar_Re, I=bar_I,
                n=m_bar['n'], c=m_bar['c']
            )
            out['bar'] = np.squeeze(downsample(bar_super, oversample_n))
        return out

    _r = jit(render, static_argnums=(1, 2, 3))
    return lambda model: _r(model, comps['bulge'], comps['bar'], n_spirals)


class Optimizer():
    def __init__(self, aggregation_result, psf, galaxy_data, sigma_image,
                 oversample_n=5, **kwargs):
        self.aggregation_result = aggregation_result
        self.model = to_reparametrization(aggregation_result.model)
        self.model_err = get_reparametrized_errors(aggregation_result)
        self.lims = get_limits(aggregation_result)
        self.psf = psf
        self.n_spirals = len(self.model['spiral']) // 6
        self.target = np.asarray(galaxy_data.data)
        self.mask = np.asarray(galaxy_data.mask)
        self.sigma = np.asarray(sigma_image.data)
        self.render = _make_render_func(
            self.model, self.target.shape, oversample_n
        )
        self.update_model_from_p = jit(self.update_model_from_p, static_argnums=(1,))
        self.step = jit(self.step, static_argnums=(1,))
        self.jacobian = jit(jacrev(self.step), static_argnums=(1,))

    def update_model_from_p(self, p, keys):
        input_model = {}
        for (k0, k1), v in zip(keys, p):
            input_model.setdefault(k0, {})
            input_model[k0][k1] = v

        return {
            k: {**self.model[k], **input_model.get(k, {})}
            for k in self.model
        }

    def step(self, p, keys):
        model_render = self.render(self.update_model_from_p(p, keys))
        rendered_gal = np.sum(list(model_render.values()), axis=0)
        psf_conv_render = psf_conv(rendered_gal, self.psf)
        param_delta = np.array([
            (i - j) / s
            for i, j, s in zip(
                p,
                get_p(self.model, keys),
                get_p(self.model_err, keys),
            )
            if np.isfinite(s)
        ])
        scaled_model_err = (
            psf_conv_render[~self.mask] - self.target[~self.mask]
        ) / self.sigma[~self.mask]
        return negative_log_likelihood(
            np.concatenate((scaled_model_err.ravel(), param_delta))
        )

    def do_fit(self, keys=None, p0=None, progress=True, desc='Fitting', **kwargs):
        keys = (
            keys if keys is not None
            else [(k0, k1) for k0 in self.model for k1 in self.model[k0]]
        )
        p0 = p0 if p0 is not None else get_p(self.model, keys)

        def _f(p, keys):
            return onp.float64(self.step(p, keys))

        def _jac(p, keys):
            return onp.array(self.jacobian(p, keys), dtype=onp.float64)

        lw_cb = kwargs.pop('callback', lambda *a, **k: None)

        if progress:
            with tqdm(desc=desc, leave=False) as pbar:
                def callback(*args, **kwargs):
                    pbar.update(1)
                    lw_cb(*args, **kwargs)
                res = minimize(
                    _f, p0, args=(keys,),
                    jac=_jac, callback=callback,
                    bounds=[self.lims[k0][k1] for k0, k1 in keys],
                    **kwargs
                )
        else:
            res = minimize(
                _f, p0, args=(keys,),
                jac=_jac, callback=lw_cb,
                bounds=[self.lims[k0][k1] for k0, k1 in keys],
                **kwargs
            )

        final_model = deepcopy(self.update_model_from_p(res['x'], keys))
        if final_model.get('bulge', None) is not None:
            final_model['bulge']['mux'] = final_model['centre']['mux']
            final_model['bulge']['muy'] = final_model['centre']['muy']
            final_model['bulge']['Re'] = (
                final_model['bulge']['scale']
                * final_model['disk']['Re']
            )
        if final_model.get('bar', None) is not None:
            final_model['bar']['mux'] = final_model['centre']['mux']
            final_model['bar']['muy'] = final_model['centre']['muy']
            final_model['bar']['Re'] = (
                final_model['bar']['scale']
                * final_model['disk']['Re']
            )

        final_model_ = deepcopy(final_model)
        for i in range(len(self.model['spiral']) // 6):
            A, phi, q, roll, (t_min, t_max) = correct_logsp_params(
                final_model['spiral']['A.{}'.format(i)],
                final_model['spiral']['phi.{}'.format(i)],
                final_model['disk']['q'],
                final_model['disk']['roll'],
                final_model['disk']['roll'] - self.model['disk']['roll'],
                onp.array((
                    final_model['spiral']['t_min.{}'.format(i)],
                    final_model['spiral']['t_max.{}'.format(i)],
                )),
            )
            final_model['spiral'].update({
                'A.{}'.format(i): A,
                'phi.{}'.format(i): phi,
                't_min.{}'.format(i): t_min,
                't_max.{}'.format(i): t_max,
            })

        return deepcopy(dict(
            base_model=self.model,
            fit_model=final_model,
            fit_model_uncorrected=final_model_,
            res=res,
            keys=keys,
        ))
