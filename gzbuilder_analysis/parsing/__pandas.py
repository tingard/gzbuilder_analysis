import numpy as np
import pandas as pd

def to_pandas(model):
    # filter out empty components
    model = {k: v for k, v in model.items() if v is not None}

    params = {
        f'{comp} {param}': model[comp][param]
        for comp in ('disk', 'bulge', 'bar')
        for param in model.get(comp, {}).keys()
    }
    params.update({
        f'spiral{i} {param}': model['spiral'][i][1][param]
        for i in range(len(model['spiral']))
        for param in model['spiral'][i][1].keys()
    })
    idx = pd.MultiIndex.from_tuples([
        k.split() for k in params.keys()
    ], names=('component', 'parameter'))
    vals = [params.get(' '.join(p), np.nan) for p in idx.values]
    return pd.Series(vals, index=idx, name='value')


def from_pandas(params, spirals=None):
    model_df = params.dropna().unstack().T
    model = dict(disk=None, bulge=None, bar=None, spiral=[])
    model.update(
        model_df.apply(lambda a: a.dropna().to_dict()).to_dict()
    )
    try:
        nspirals = max(int(i.replace('spiral', '')) for i in model_df.columns if 'spiral' in i) + 1
        if spirals is not None:
            if len(spirals) != nspirals:
                raise ValueError(
                    'Mismatch in spiral count between points and parameters'
                )
            model['spiral'] = [
                (np.array(spirals[i]), model.pop(f'spiral{i}'))
                for i in range(nspirals)
            ]
        else:
            model['spiral'] = [
                (np.array([]), model.pop(f'spiral{i}'))
                for i in range(nspirals)
            ]
    except ValueError:
        pass
    return model
