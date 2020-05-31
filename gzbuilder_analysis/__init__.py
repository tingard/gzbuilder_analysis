import os
import re
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


def load_aggregation_results(path='output_files/aggregation_results', ext='pkl.gz'):
    """Given a directory of aggregation results, read all appropriate files
    """
    return __read_files(
        path,
        ext,
        pbar_kw=dict(desc='Loading aggregation results'),
    ).apply(pd.Series)
    return agg_results


def load_fit_results(path='output_files/tuning_results', ext='pkl.gz', include_bad=False):
    """Given a directory of fit results, read all appropriate files,
    filtering for bad fits.
    """
    fit_models = __read_files(
        path,
        ext,
        pbar_kw=dict(desc='Loading aggregation results'),
    )
    if not include_bad:
        return fit_models.apply(
            lambda m: m if m['res']['success'] else np.nan
        ).dropna().apply(pd.Series)
    return fit_models.apply(pd.Series)


def to_catalog(fit_results, fitting_metadata):
    """TODO:
    """
    pass
