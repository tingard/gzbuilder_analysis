import numpy as np
import pandas as pd


def jaccard_distance(ob1, ob2):
    if ob1.union(ob2).area <= 0:
        return 1
    return 1 - ob1.intersection(ob2).area / ob1.union(ob2).area


def make_jaccard_distances(geoms):
    distances = pd.DataFrame([], columns=geoms.index)
    for i, (idx, g) in enumerate(geoms.iteritems()):
        d = np.zeros(len(geoms.index))
        for j in range(i + 1, len(geoms)):
            d[j] = jaccard_distance(geoms[idx], geoms.iloc[j])
        distances.loc[idx] = d
    return distances + distances.T
