import numpy as np
import pandas as pd
from gzbuilder_analysis.parsing import to_pandas, downsample
from gzbuilder_analysis.config import DEFAULT_SPIRAL
from .spirals import get_drawn_arms
from .spirals.oo import Pipeline
from .__cluster import cluster_components
from .__aggregate import circular_error, aggregate_components


# probably doesn't need to be a class, given it has no methods.
class AggregationResult(object):
    def __init__(self, models, galaxy_data):
        self.clusters = cluster_components(
            models=models,
            image_size=galaxy_data.shape,
            warn=False
        )
        self.input_models = models
        self.aggregation_result = aggregate_components(self.clusters)
        self.phi = self.aggregation_result['disk']['roll']
        self.ba = self.aggregation_result['disk']['q']
        self.centre_pos = np.array((
            self.aggregation_result['disk']['mux'],
            self.aggregation_result['disk']['muy']
        ))
        drawn_arms = get_drawn_arms(models, min_n=5)
        if len(drawn_arms) > 0:
            self.spiral_pipeline = Pipeline(
                drawn_arms.values,
                centre_pos=self.centre_pos,
                phi=self.phi, ba=self.ba,
                image_size=galaxy_data.shape
            )
            self.spiral_arms = self.spiral_pipeline.get_arms()
        else:
            self.spiral_pipeline = None
            self.spiral_arms = []
        self.model = aggregate_components(self.clusters)
        self.model['spiral'] = [
            (downsample(a.reprojected_log_spiral), DEFAULT_SPIRAL)
            for a in self.spiral_arms
        ]
        self.errors = pd.Series({
            comp: self.clusters[comp].apply(pd.Series).std()
            for comp in ('disk', 'bulge', 'bar')
        }).apply(pd.Series).stack().rename_axis(('component', 'parameter'))
        for comp in ('disk', 'bulge', 'bar'):
            if self.model[comp] is not None:
                self.errors[comp, 'roll'] = circular_error(
                    self.clusters[comp].apply(lambda m: m['roll']).values,
                    2
                )[1]
        unconstrained_errs = pd.concat((
            self.errors.xs('I', level=1, drop_level=False),
            self.errors.xs('n', level=1, drop_level=False),
            self.errors.xs('c', level=1, drop_level=False),
        ))
        self.errors[unconstrained_errs.index] = np.inf
        self.params = to_pandas(self.model).rename('model')
