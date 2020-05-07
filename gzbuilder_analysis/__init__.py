import os
import re
import pandas as pd
from tqdm import tqdm


def df_to_dict(df):
    """Quickly convert a DataFrame to a dictionary, removing any NaNs
    """
    return {k: v[v.notna()].to_dict() for k, v in df.items()}


def load_aggregation_results(path='output_files/aggregation_results'):
    agg_results = pd.Series([], dtype=object)
    with tqdm(
        os.listdir(path),
        desc='Loading aggregation results',
        leave=False
    ) as bar:
        for f in bar:
            if re.match(r'[0-9]+\.pkl.gz', f):
                agg_results[int(f.split('.')[0])] = pd.read_pickle(
                    os.path.join(path, f)
                )
    return agg_results


def load_fit_results(path='output_files/tuning_results', include_bad=False):
    fit_models = pd.Series([], dtype=object)
    with tqdm(
        os.listdir(path),
        desc='Loading fitting results',
        leave=False
    ) as bar:
        for f in bar:
            if re.match(r'[0-9]+\.pickle.gz', f):
                fit_result = pd.read_pickle(
                    os.path.join(path, f)
                )
                if fit_result['res']['success'] or include_bad:
                    fit_models[int(f.split('.')[0])] = fit_result
    return fit_models.apply(pd.Series)


def to_catalog(fit_results, fitting_metadata):
    pass
