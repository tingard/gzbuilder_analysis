import numpy as np
import pandas as pd
from gzbuilder_analysis.parsing import to_pandas, downsample
from gzbuilder_analysis.config import DEFAULT_SPIRAL
from .spirals import get_drawn_arms
from .spirals.oo import Pipeline
from .__cluster import cluster_components
from .__aggregate import aggregate_components


class AggregationResult(object):
    def __init__(self, models, galaxy_data):
        self.clusters = cluster_components(
            models=models,
            image_size=galaxy_data.shape,
            warn=False
        )
        self.aggregation_result = aggregate_components(self.clusters)
        self.phi = self.aggregation_result['disk']['roll']
        self.ba = self.aggregation_result['disk']['q']
        self.centre_pos = np.array((
            self.aggregation_result['disk']['mux'],
            self.aggregation_result['disk']['muy']
        ))
        drawn_arms = get_drawn_arms(models, min_n=5)
        self.__spiral_pipeline = Pipeline(
            drawn_arms.values,
            centre_pos=self.centre_pos,
            phi=self.phi, ba=self.ba,
            image_size=galaxy_data.shape
        )
        if self.__spiral_pipeline is not None:
            self.spiral_arms = self.__spiral_pipeline.get_arms()
        else:
            self.__spiral_arms = []
        self.__model = aggregate_components(self.clusters)
        self.__model['spiral'] = [
            (downsample(a.reprojected_log_spiral), DEFAULT_SPIRAL)
            for a in self.spiral_arms
        ]
        self.errors = pd.Series({
            comp: self.clusters[comp].apply(pd.Series).std()
            for comp in ('disk', 'bulge', 'bar')
        }).apply(pd.Series).stack().rename_axis(('component', 'parameter'))
        self.params = to_pandas(self.__model).rename('model')

    def __calculate_spirals(self):
        kw = dict(
            centre=self.params[[('disk', 'mux'), ('disk', 'muy')]],
            phi=self.params[('disk', 'roll')],
            ba=self.params[('disk', 'q')]
        )
        [arm.modify_disk(**kw) for arm in self.spiral_arms]

    # used during fitting
    def update_params(self, new_params):
        delta = (self.params - new_params).dropna()
        self.params.update(new_params)
        if np.any(delta.xs('disk', level=0) == 0):
            self.__calculate_spirals()
            return True
        return False

    def update_disk(self, new_disk):
        """When the disk is modified, we may need to recalculate spiral arms
        """
        new_disk = pd.DataFrame(
            {'disk': new_disk}
        ).unstack().reindex_like(self.params)
        delta = (self.params - new_disk).dropna()
        if np.any(delta != 0.0):
            self.params.update(new_disk)
            self.__calculate_spirals()
