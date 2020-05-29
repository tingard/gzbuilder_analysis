from functools import wraps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.affinity import scale, translate
from descartes import PolygonPatch
from astropy.visualization import AsinhStretch
from gzbuilder_analysis.aggregation import get_geoms, make_ellipse, make_box
from gzbuilder_analysis.aggregation.spirals import get_drawn_arms
from gzbuilder_analysis import df_to_dict
from gzbuilder_analysis.rendering import psf_conv
from gzbuilder_analysis.rendering import get_spirals


_stretch = AsinhStretch()


def gen_utils(data):
    def transform_patch(p, s=2):
        corrected_patch = scale(
            translate(p, xoff=-data.shape[1]/2, yoff=-data.shape[0]/2),
            xfact=0.396,
            yfact=0.396,
            origin=(0, 0),
        )
        # display patch at s*Re
        return scale(corrected_patch, s, s)

    def transform_arm(arm):
        return (arm - np.array(data.shape) / 2)* 0.396

    extent = (np.array([[-1, -1],[1, 1]]) * data.shape).T.ravel() / 2 * 0.396
    imshow_kwargs = {
        'cmap': 'gray_r',
        'origin': 'lower',
        'extent': extent
    }
    return pd.Series(dict(
        transform_patch=transform_patch,
        transform_arm=transform_arm,
        extent=extent,
        imshow_kwargs=imshow_kwargs
    ))


def plot_models(models, data, **kwargs):
    utils = gen_utils(data)
    geoms = pd.DataFrame(
        models.apply(get_geoms).values.tolist(),
        columns=('disk', 'bulge', 'bar')
    )
    drawn_arms = get_drawn_arms(models)

    fig, ax = plt.subplots(
        ncols=2, nrows=2,
        figsize=kwargs.pop('figsize', (8, 8)),
        dpi=kwargs.pop('dpi', 100),
        sharex=True, sharey=True
    )
    [a.imshow(_stretch(data), **utils.imshow_kwargs) for a in ax.ravel()]

    ((ax0, ax1), (ax2, ax3)) = ax
    for comp in geoms['disk'].values:
        if comp is not None:
            ax0.add_patch(
                PolygonPatch(utils.transform_patch(comp), fc='none', ec='k',
                             zorder=3)
            )
    for comp in geoms['bulge'].values:
        if comp is not None:
            ax1.add_patch(
                PolygonPatch(utils.transform_patch(comp), fc='none', ec='k',
                             zorder=3)
            )
    for comp in geoms['bar'].values:
        if comp is not None:
            ax2.add_patch(
                PolygonPatch(utils.transform_patch(comp), fc='none', ec='k',
                             zorder=3)
            )
    for arm in drawn_arms:
        ax3.plot(*utils.transform_arm(arm).T, 'k', alpha=0.5, linewidth=1)

    for i, ax in enumerate((ax0, ax1, ax2, ax3)):
        ax.set_xlim(utils.imshow_kwargs['extent'][:2])
        ax.set_ylim(utils.imshow_kwargs['extent'][2:])
        if i % 2 == 0:
            ax.set_ylabel('Arcseconds from center')
        if i > 1:
            ax.set_xlabel('Arcseconds from center')
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()
    return fig, ax


def plot_aggregation_result(aggregation_result, data, **kwargs):
    utils = gen_utils(data)
    disk_crop = min(np.abs(utils.imshow_kwargs['extent']).min(), 1000)
    bulge_crop = bar_crop = min(np.abs(utils.imshow_kwargs['extent']).min(), 15)
    spiral_crop = min(np.abs(utils.imshow_kwargs['extent']).min(), 25)

    geoms = get_geoms(aggregation_result.model)
    def make_patches(c=('C0', 'C1', 'C2'), ls=('-.', ':', '--'), **kwargs):
        k = { 'alpha': 0.3, 'zorder': 3, 'ec': 'k', **kwargs}
        patches = [
            PolygonPatch(
                utils.transform_patch(geoms[comp]),
                fc=c[i],
                linestyle=ls[i],
                **k,
            ) if geoms.get(comp, None) is not None else None
            for i, comp in enumerate(
                ('disk', 'bulge', 'bar')
            )
        ]
        return patches
    edge_patches = make_patches(
        c=('none', 'none', 'none'),
        alpha=1,
        lw=2,
    )
    face_patches = make_patches(
        ec='none',
    )
    fig = plt.figure(
        figsize=kwargs.pop('figsize', (8, 8)),
        dpi=kwargs.pop('dpi', 100)
    )
    ax = plt.gca()
    plt.imshow(_stretch(data), **utils.imshow_kwargs)

    try:
        plt.gca().add_patch(edge_patches[0])
        plt.gca().add_patch(face_patches[0])
    except AttributeError:
        pass

    try:
        plt.gca().add_patch(edge_patches[2])
        plt.gca().add_patch(face_patches[2])
    except AttributeError:
        pass

    try:
        plt.gca().add_patch(edge_patches[1])
        plt.gca().add_patch(face_patches[1])
    except AttributeError:
        pass

    for arm in aggregation_result.spiral_arms:
        a = utils.transform_arm(arm.reprojected_log_spiral)
        plt.plot(*a.T, 'r', lw=2, zorder=4)

    plt.xlabel('Arcseconds from galaxy centre')
    plt.ylabel('Arcseconds from galaxy centre')
    plt.xlim(-disk_crop, disk_crop)
    plt.ylim(-disk_crop, disk_crop)

    plt.tight_layout()
    return fig, ax


def plot_tuning_result(data, aggregation_result, final_model, deparametrized_model, comps, psf, sigma_image, **kwargs):
    utils = gen_utils(data)
    # deparametrized_model = df_to_dict(deparametrized_model.unstack().T)
    final_gal = np.ma.masked_array(psf_conv(sum(comps.values()), psf), data.mask)
    n_spirals = len(final_model.get('spiral', {})) // 6
    fit_spirals = get_spirals(final_model.to_dict(), n_spirals, final_model.disk.roll)
    lm = _stretch([
        min(np.nanmin(data), np.nanmin(final_gal)),
        max(np.nanmax(data), np.nanmax(final_gal)),
    ])
    d = (final_gal - data) / sigma_image
    l2 = np.nanmax(np.abs(d))

    fig, ax = plt.subplots(
        nrows=2, ncols=3,
        figsize=kwargs.pop('figsize', (15*1.8, 8.*1.8)),
        dpi=kwargs.pop('dpi', 100),
    )
    plt.subplots_adjust(wspace=0, hspace=0.15)
    [
        a.imshow(_stretch(data), vmin=lm[0], vmax=lm[1], **utils.imshow_kwargs)
        for a in (ax[0, 0], ax[1, 0], ax[1, 1])
    ]
    ax[0, 1].imshow(
        _stretch(final_gal),
        vmin=lm[0], vmax=lm[1], **utils.imshow_kwargs,
    )
    c = ax[0][2].imshow(d, vmin=-l2, vmax=l2, cmap='seismic', origin='lower')
    cbar = plt.colorbar(c, ax=ax, shrink=0.475, anchor=(0, 1))
    cbar.ax.set_ylabel(r'Residual, units of $\sigma$')

    # add component overlays
    initial_disk = make_ellipse(aggregation_result.model['disk'])
    final_disk = make_ellipse(deparametrized_model['disk'])

    ax[1, 0].add_patch(PolygonPatch(utils.transform_patch(initial_disk), ec='C0', fc='none'))
    ax[1, 1].add_patch(PolygonPatch(utils.transform_patch(final_disk), ec='C0', fc='none'))

    if aggregation_result.model.get('bulge', None) is not None:
        initial_bulge = make_ellipse(aggregation_result.model['bulge'])
        ax[1, 0].add_patch(
            PolygonPatch(
                utils.transform_patch(initial_bulge),
                ec='C1', fc='none'
            )
        )
    if deparametrized_model.get('bulge', None) is not None:
        final_bulge = make_ellipse(deparametrized_model['bulge'])
        ax[1, 1].add_patch(PolygonPatch(utils.transform_patch(final_bulge), ec='C1', fc='none'))

    if aggregation_result.model.get('bar', None) is not None:
        initial_bar = make_box(aggregation_result.model['bar'])
        ax[1, 0].add_patch(PolygonPatch(utils.transform_patch(initial_bar), ec='C2', fc='none'))
    if deparametrized_model.get('bar', None) is not None:
        final_bar = make_box(deparametrized_model['bar'])
        ax[1, 1].add_patch(PolygonPatch(utils.transform_patch(final_bar), ec='C2', fc='none'))

    for arm in aggregation_result.spiral_arms:
        ax[1, 0].plot(*utils.transform_arm(arm.reprojected_log_spiral).T, 'r')

    for arm in fit_spirals:
        ax[1, 1].plot(*utils.transform_arm(arm).T, 'r')
    ax[0, 0].set_title('Galaxy Image')
    ax[0, 1].set_title('Model after fitting')
    ax[0, 2].set_title('Residuals')
    ax[1, 0].set_title('Raw Aggregate model\noverlaid on galaxy')
    ax[1, 1].set_title('Fit model overlaid on galaxy\n')
    for a in ax.ravel():
        a.axis('off')
    return fig, ax
