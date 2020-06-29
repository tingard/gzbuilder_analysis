import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


def df_to_dict(df):
    """Quickly convert a DataFrame to a dictionary, removing any NaNs
    """
    return {k: v[v.notna()].to_dict() for k, v in df.items()}


def __read_files(path, ext, pbar_kw={}):
    """Given a directory, read all appropriate files
    """
    res = pd.Series([], dtype=object)
    with tqdm(
        os.listdir(path),
        leave=pbar_kw.pop('leave', False),
        **pbar_kw,
    ) as bar:
        for f in bar:
            if re.match(r'[0-9]+\.{}'.format(ext), f):
                res[int(f.split('.')[0])] = pd.read_pickle(
                    os.path.join(path, f)
                )
    return res


def load_aggregation_results(
    path='output_files/aggregation_results',
    ext='pkl.gz'
):
    """Given a directory of aggregation results, read all appropriate files
    """
    return __read_files(
        path,
        ext,
        pbar_kw=dict(desc='Loading aggregation results'),
    )


def load_fit_results(
    path='output_files/tuning_results',
    ext='pkl.gz',
    include_bad=False
):
    """Given a directory of fit results, read all appropriate files,
    filtering for bad fits.
    """
    fit_models = __read_files(
        path,
        ext,
        pbar_kw=dict(desc='Loading aggregation results'),
    )
    if not include_bad:
        fit_models = fit_models.apply(
            lambda m: m if m['res']['success'] else None
        ).dropna()
    return fit_models.apply(pd.Series)


def __to_pogson_magnitude(L):
    return 22.5 - 2.5 * np.log10(L)


def to_catalog(fit_results, aggregation_results, zoo_subject_data,
               validation_set_ids=[]):
    """Given a set of aggregate models, fit models and fitting metadata,
    compile a catalog to be saved and published.

    # Columns
    - dr7objid
    - is_validation (is the model a repeat?)
    - ra
    - dec
    - chisq # the reduced chi-square value of the fit
    - pogson_magnitude # r-band model pogson magnitude
    - flux # r-band model flux
    - disc_Re # disc effective radius
    - disc_q # disc ellipticity
    - centre_dx # position offset, arcseconds
    - centre_dy # position offset, arcseconds
    - bulge_Re
    - bulge_q
    - bulge_n
    - bulge_fraction # the relative contribution of the bulge to the total
                       galaxy flux
    - bar_Re
    - bar_q
    - bar_n
    - bar_c
    - bar_fraction
    - n_spirals
    - spiral_pa_0 # pitch angle of 0th spiral arm
    ...
    - spiral_pa_n # pitch angle of nth spiral arm
    - spiral_fraction # the relative contribution of the spirals to the total
                        galaxy flux

    We recommend that a catalog of cutout data, pixel masks, sigma images and
    rendered component values are published separately
    """
    subject_data = zoo_subject_data \
        .reset_index(drop=False) \
        .drop_duplicates(subset='subject_id') \
        .set_index('subject_id')

    metadata = subject_data['metadata'] \
        .apply(json.loads) \
        .apply(pd.Series)

    dr7objid = metadata['SDSS dr7 id'] \
        .dropna() \
        .astype(np.int64) \
        .replace(-1, np.nan) \
        .rename('dr7objid')

    is_validation = subject_data \
        .eval('subject_set_id in @validation_set_ids') \
        .rename('is_validation')

    gal_magnitude = __to_pogson_magnitude(
        fit_results['r_band_luminosity']
    ).rename('pogson_magnitude')
    fit_models = fit_results['fit_model'] \
        .apply(pd.Series)
    deparametrized_fit_models = fit_results['deparametrized'].apply(pd.Series)
    spiral_luminosity = fit_results['comps'] \
        .apply(pd.Series)['spiral'] \
        .dropna() \
        .apply(np.sum)
    spiral_frac = (spiral_luminosity / fit_results['r_band_luminosity']) \
        .rename('spiral_fraction')
    central_offset = (
        fit_models[[('disk', 'mux'), ('disk', 'muy')]].droplevel(0, axis=1)
        - fit_models[[('centre', 'mux'), ('centre', 'muy')]].droplevel(0, axis=1)
    ).rename(columns=dict(mux='centre_dx', muy='centre_dy'))

    spirals = fit_models['spiral']
    spirals.columns = pd.MultiIndex.from_tuples([i.split('.') for i in spirals.columns])

    aggregation_uncertainty = aggregation_results.apply(lambda a: a.errors)
    n_spirals = spirals['A'].apply(lambda a: len(a.dropna()), axis=1) \
        .astype(int) \
        .rename('n_spirals')
    catalogue = pd.concat((
        dr7objid,
        is_validation,
        metadata.loc[dr7objid.index][['ra', 'dec']].astype(float),
        fit_results['chisq'],
        gal_magnitude,
        fit_results['r_band_luminosity'].rename('flux'),
        fit_models[('disk', 'Re')].rename('disk_Re'),
        aggregation_uncertainty[('disk', 'Re')].rename('disk_Re_err'),
        fit_models[('disk', 'q')].rename('disk_q'),
        aggregation_uncertainty[('disk', 'q')].rename('disk_q_err'),
        central_offset,
        deparametrized_fit_models[('bulge', 'Re')].rename('bulge_Re'),
        aggregation_uncertainty[('bulge', 'q')].rename('bulge_Re_err'),
        fit_models[('bulge', 'q')].rename('bulge_q'),
        aggregation_uncertainty[('bulge', 'q')].rename('bulge_q_err'),
        fit_models[('bulge', 'n')].rename('bulge_n'),
        fit_results['bulge_frac'].replace(0, np.nan).rename('bulge_fraction'),
        deparametrized_fit_models[('bar', 'Re')].rename('bar_Re'),
        aggregation_uncertainty[('bar', 'q')].rename('bar_Re_err'),
        fit_models[('bar', 'q')].rename('bar_q'),
        aggregation_uncertainty[('bar', 'q')].rename('bar_q_err'),
        fit_models[('bar', 'n')].rename('bar_n'),
        fit_models[('bar', 'c')].rename('bar_c'),
        fit_results['bar_frac'].replace(0, np.nan).rename('bar_fraction'),
        spirals['phi'].add_prefix('pitch_angle_'),
        n_spirals,
        spiral_frac,
    ), axis=1)
    catalogue.index.name = 'galaxy_builder_id'
    catalogue = catalogue \
        .reset_index(drop=False) \
        .set_index('dr7objid')
    return catalogue
