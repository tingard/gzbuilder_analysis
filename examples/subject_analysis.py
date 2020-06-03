import os
import requests
import json
import re
import tempfile
import queue
import warnings
from datetime import datetime
from time import sleep
from copy import deepcopy
from io import BytesIO
from getpass import getpass
import shutil
from PIL import Image
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import montage_wrapper as montage
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.io import fits
from astropy.wcs import WCS
from astropy import log
from astropy import log
from panoptes_client import Panoptes, Project

from gzbuilder_analysis.data import get_sdss_cutout, download_json, download_image, \
    get_montage_cutout, get_rotation_correction
import gzbuilder_analysis.parsing as pg
import gzbuilder_analysis.aggregation as ag
from gzbuilder_analysis.parsing.reparametrization import from_reparametrization
from gzbuilder_analysis.fitting.optimizer import Optimizer
from gzbuilder_analysis.rendering import psf_conv, render_comps
from gzbuilder_analysis.fitting.misc import get_luminosity_keys, \
    remove_zero_brightness_components, lower_spiral_indices, correct_spirals, \
    correct_axratio

from gzbuilder_analysis.plotting import plot_models, plot_aggregation_result, \
    plot_tuning_result

import jax.numpy as jnp
from jax import ops
from jax.config import config
config.update("jax_enable_x64", True)

# We need a Zooniverse classification and subject export
try:
    classifications = pd.read_csv('galaxy-builder-classifications.csv', index_col=0)
    subjects = pd.read_csv('galaxy-builder-subjects.csv', index_col=0)
except FileNotFoundError:
    username = input('Enter username')
    password = getpass('Enter Password')
    Panoptes.connect(username=username, password=password)
    del password

    PROJECT_SLUG = 'tingard/galaxy-builder'
    proj = Project.find(slug=PROJECT_SLUG)

    # Generate exports and parse into DataFrames:
    classification_export = proj.get_export('classifications', generate=False)
    classifications = pd.DataFrame([i for i in classification_export.csv_dictreader()]) \
        .set_index('classification_id') \
        .astype({'subject_ids': int, 'created_at': 'datetime64[s]'})
    # convert the index dtype from Object to int
    classifications.index = classifications.index.astype(int)
    classifications.to_csv('galaxy-builder-classifications.csv')

    subject_export = proj.get_export('subjects', generate=False)

    # restrict to galaxies after the beta, and not in the calibration set, and
    # make sure each subject only appears once
    subjects = pd.DataFrame([i for i in subject_export.csv_dictreader()]) \
        .astype({'subject_set_id': int}) \
        .query('subject_set_id >= 20561 and subject_set_id != 80112') \
        .groupby('subject_id') \
        .apply(lambda a: a.iloc[-1]) \
        .set_index('subject_id')
    subjects.index = subjects.index.astype(int)
    subjects.to_csv('galaxy-builder-subjects.csv')

metadata = subjects.pop('metadata').apply(json.loads).apply(pd.Series)
locations = subjects.pop('locations').apply(json.loads) \
    .apply(pd.Series) \
    .rename(columns={'0': 'difference', '1': 'image', '2': 'model'})

bands = ['r']


def do_subject(subject_id):

    ###### Subject Preparation ######
    # get the galaxy position from its metadata
    pos = metadata.loc[subject_id][
        ['ra', 'dec', 'Petrosian radius (degrees)']
    ].astype(float)
    with tqdm.tqdm(desc='Downloading Zooniverse data', leave=False) as bar:
        fitting_metadata = download_json(
            locations['difference'].loc[subject_id]
        ).apply(np.array)
        zoo_image = download_image(locations['image'].loc[subject_id])

    # create the stacked image from SDSS frames
    with tqdm.tqdm(desc='Downloading SDSS frames', leave=False) as bar:
        cutout_data, frame_data = get_sdss_cutout(
            pos['ra'],
            pos['dec'],
            cutout_radius=pos['Petrosian radius (degrees)'],
            bands=bands,
            return_frame_data=True
        )

    # due to complications during project design, we need to map from the
    # uploaded image (created using montage, which has a smoothing effect),
    # to the properly stacked image here
    with tqdm.tqdm(desc='Performing montage', leave=False) as bar:
        montage_cutout = get_montage_cutout(
            frame_data,
            pos['ra'],
            pos['dec'],
            pos['Petrosian radius (degrees)']
        )
    montage_wcs = montage_cutout.wcs
    # some images show a weird rotation, we account for this by calculating
    # the optimal residual between the montaged result and the image shown
    # to volunteers
    rotation_correction = get_rotation_correction(
        montage_cutout.data,
        fitting_metadata['imageData'],
        fitting_metadata['mask'].astype(bool)
    )

    target_wcs = cutout_data[bands[0]]['wcs']

    data = cutout_data['r']['data']

    ###### Parsing input models ######
    c = (classifications
        .query('subject_ids == {}'.format(subject_id))
        .sort_values(by='created_at')
        .head(30)
    )

    size_diff = zoo_image.size[0] / data.shape[0]
    zoo_models = c.apply(
        pg.parse_classification,
        axis=1,
        image_size=np.array(zoo_image.size),
        size_diff=size_diff,
        ignore_scale=True  # ignore scale slider when aggregating
    )
    scaled_models = zoo_models.apply(
        pg.scale_model,
        args=(1/size_diff,),
    )
    rotated_models = scaled_models.apply(
        pg.rotate_model_about_centre,
        args=(
            data.shape[0],
            rotation_correction
        ),
    )
    models = rotated_models.apply(
        pg.reproject_model,
        wcs_in=montage_wcs, wcs_out=target_wcs
    )

    sanitized_models = models.apply(pg.sanitize_model)

    ###### Clustering ######
    try:
        aggregation_result = ag.AggregationResult(
            sanitized_models,
            cutout_data['r']['data']
        )
    except TypeError as e:
        log.warn('No disk cluster for {}'.format(subject_id))
        raise(e)

    ###### Optimization ######
    # define two handy functions to read results back from the GPU for scipy's
    # LBFGS-b
    def __f(p, optimizer, keys):
        return np.array(optimizer(p, keys).block_until_ready(), dtype=np.float64)


    def __j(p, optimizer, keys):
        return np.array(optimizer.jac(p, keys).block_until_ready(), dtype=np.float64)


    def __bar_incrementer(bar):
        def f(*a, **k):
            bar.update(1)
        return f


    o = Optimizer(
        aggregation_result,
        cutout_data['r']['psf'].astype(np.float64),
        cutout_data['r']['data'].astype(np.float64),
        cutout_data['r']['sigma'].astype(np.float64),
        oversample_n=5
    )
    # define the parameters controlling only the brightness of components, and
    # fit them first
    L_keys = get_luminosity_keys(o.model)

    # perform the first fit
    with tqdm.tqdm(desc='Fitting brightness', leave=False) as bar:
        res = minimize(
            __f,
            np.array([o.model_[k] for k in L_keys]),
            jac=__j,
            args=(o, L_keys),
            callback=__bar_incrementer(bar),
            bounds=np.array([o.lims_[k] for k in L_keys]),
        )

    # update the optimizer with the new parameters
    for k, v in zip(L_keys, res['x']):
        o[k] = v

    # perform the full fit
    with tqdm.tqdm(desc='Fitting everything', leave=False) as bar:
        res_full = minimize(
            __f,
            np.array([o.model_[k] for k in o.keys]),
            jac=__j,
            args=(o, o.keys),
            callback=__bar_incrementer(bar),
            bounds=np.array([o.lims_[k0][k1] for k0, k1 in o.keys]),
            options=dict(maxiter=10000)
        )

    ###### Cleanup ######
    final_model = pd.Series({
        **deepcopy(o.model_),
        **{k: v for k, v in zip(o.keys, res_full['x'])}
    })

    # correct the parameters of spirals in this model for the new disk,
    # allowing rendering of the model without needing the rotation of the disk
    # before fitting
    final_model = correct_spirals(final_model, o.base_roll)

    # fix component axis ratios (if > 1, flip major and minor axis)
    final_model = correct_axratio(final_model)

    # remove components with zero brightness
    final_model = remove_zero_brightness_components(final_model)

    # lower the indices of spirals where possible
    final_model = lower_spiral_indices(final_model)

    comps = o.render_comps(final_model.to_dict(), correct_spirals=False)

    d = ops.index_update(
        psf_conv(sum(comps.values()), o.psf) - o.target,
        o.mask,
        np.nan
    )
    chisq = float(np.sum((d[~o.mask] / o.sigma[~o.mask])**2) / (~o.mask).sum())
    disk_spiral_L = (
        final_model[('disk', 'L')]
        + (comps['spiral'].sum() if 'spiral' in comps else 0)
    )
    # fractions were originally parametrized vs the disk and spirals (bulge
    # had no knowledge of bar and vice versa)
    bulge_frac = final_model.get(('bulge', 'frac'), 0)
    bar_frac = final_model.get(('bar', 'frac'), 0)

    bulge_L = bulge_frac * disk_spiral_L / (1 - bulge_frac)
    bar_L = bar_frac * disk_spiral_L / (1 - bar_frac)
    gal_L = disk_spiral_L + bulge_L + bar_L

    bulge_frac = bulge_L / (disk_spiral_L + bulge_L + bar_L)
    bar_frac = bar_L / (disk_spiral_L + bulge_L + bar_L)

    deparametrized_model = from_reparametrization(final_model, o)

    ftol = 2.220446049250313e-09

    # Also calculate Hessian-errors
    errs = np.sqrt(
        max(1, abs(res_full.fun))
        * ftol
        * np.diag(res_full.hess_inv.todense())
    )


    ###### Plotting ######
    plot_models(data, sanitized_models, dpi=80)
    os.makedirs('images/volunteer_models/png', exist_ok=True)
    plt.savefig('images/volunteer_models/{}.pdf'.format(subject_id), bbox_inches='tight')
    plt.savefig('images/volunteer_models/png/{}.png'.format(subject_id), bbox_inches='tight')
    plt.close()

    plot_aggregation_result(data, aggregation_result, figsize=(6, 6), dpi=80)
    os.makedirs('images/aggregate_models/png', exist_ok=True)
    plt.savefig('images/aggregate_models/{}.pdf'.format(subject_id), bbox_inches='tight')
    plt.savefig('images/aggregate_models/png/{}.png'.format(subject_id), bbox_inches='tight')
    plt.close()

    plot_tuning_result(data, aggregation_result, final_model, deparametrized_model, comps, o.psf, o.sigma)
    os.makedirs('images/tuning_result/png', exist_ok=True)
    plt.savefig('images/tuning_result/{}.pdf'.format(subject_id), bbox_inches='tight')
    plt.savefig('images/tuning_result/png/{}.png'.format(subject_id), bbox_inches='tight')
    plt.close()

    ###### Return complied value ######
    return dict(
        aggregation_result=aggregation_result,
        aggregate_errors=aggregation_result.errors,
        final_model=final_model,
        deparametrized_model=deparametrized_model,
        res=res_full,
        chisq=chisq,
        comps=comps,
        r_band_luminosity=float(gal_L),
        bulge_frac=float(bulge_frac),
        bar_frac=float(bar_frac),
        hessian_errors=errs,
        keys=o.keys,
        subject=subjects.loc[subject_id],
        montage_wcs=montage_wcs,
        target_wcs=target_wcs,
        data=data,
        sigma_image=cutout_data['r']['sigma'].astype(np.float64),
        psf=cutout_data['r']['psf'].astype(np.float64),
    )


def main(overwrite=False):
    os.makedirs('aggregation_results', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    q = queue.Queue()
    [q.put(subject_id) for subject_id in subjects.index.values]
    with tqdm.tqdm(desc='iterating_over_subjects') as bar:
        while not q.empty():
            subject_id = q.get()
            if not overwrite and os.path.exists('results/{}.pickle.gz'.format(subject_id)):
                sleep(0.025)
            else:
                try:
                    result = do_subject(subject_id)
                    agg_res = result.pop('aggregation_result', None)
                    pd.to_pickle(agg_res, 'aggregation_results/{}.pickle.gz'.format(subject_id))
                    pd.to_pickle(result, 'results/{}.pickle.gz'.format(subject_id))
                except Exception as e:
                    log.warn((subject_id, e))
                    q.put(subject_id)
            bar.update(1)

    # for subject_id in tqdm.tqdm(subjects.index, desc='iterating_over_subjects'):
    #     if not overwrite and os.path.exists('results/{}.pickle.gz'.format(subject_id)):
    #         sleep(0.1)
    #         continue
    #     result = do_subject(subject_id)
    #     if result is not None:
    #         agg_res = result.pop('aggregation_result', None)
    #         pd.to_pickle(agg_res, 'aggregation_results/{}.pickle.gz'.format(subject_id))
    #         pd.to_pickle(result, 'results/{}.pickle.gz'.format(subject_id))


if __name__ == '__main__':
    # suppress warnings to keep our beautiful progress bar working
    warnings.simplefilter('ignore', UserWarning)
    log.setLevel('WARN')
    np.seterr(divide='ignore', invalid='ignore')

    os.makedirs('logs', exist_ok=True)
    with log.log_to_file('logs/{}.log'.format(str(datetime.now()).replace(' ', '_'))):
        main()
