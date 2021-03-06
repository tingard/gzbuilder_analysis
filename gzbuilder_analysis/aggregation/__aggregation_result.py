import numpy as np
import pandas as pd
from gzbuilder_analysis.parsing import to_pandas, downsample
from gzbuilder_analysis.config import COMPONENT_CLUSTERING_PARAMS, DEFAULT_SPIRAL
from .spirals import get_drawn_arms, inclined_log_spiral
from .spirals.oo import Pipeline
from .__cluster import cluster_components
from .__aggregate import circular_error, aggregate_components


class AggregationResult(object):
    """Handy class which accepts a Series of Galaxy Builder parsed models, and
    performs Aggregation of all components, including error calculation

    Arguments:
    models -- the models to aggregate
    galaxy_data -- the numpy array of pixel values for the galaxy (used for
                   scale normalization)

    Keyword Arguments:
    clustering_params -- The clustering parameters to use (see
                         gzbuilder_analysis.config.COMPONENT_CLUSTERING_PARAMS)
    """
    def __init__(self, models, galaxy_data,
                 clustering_params=COMPONENT_CLUSTERING_PARAMS):
        self.clusters = cluster_components(
            models=models,
            image_size=galaxy_data.shape,
            params=clustering_params,
            warn=False
        )
        self.input_models = models
        self._model_dict = aggregate_components(
            self.clusters,
            spiral_merging_distance=clustering_params['spiral']['merging_distance']
        )
        self.phi = self._model_dict['disk']['roll']
        self.ba = self._model_dict['disk']['q']
        self.centre_pos = np.array((
            self._model_dict['disk']['mux'],
            self._model_dict['disk']['muy']
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
        # self._model_dict = aggregate_components(self.clusters)
        spirals = [
            (downsample(a.reprojected_log_spiral), DEFAULT_SPIRAL)
            for a in self.spiral_arms
        ]
        self.errors = pd.Series({
            comp: self.clusters[comp].apply(pd.Series).std()
            for comp in ('disk', 'bulge', 'bar')
        }).apply(pd.Series).stack().rename_axis(('component', 'parameter'))
        for comp in ('disk', 'bulge', 'bar'):
            if self._model_dict[comp] is not None:
                self.errors[comp, 'roll'] = circular_error(
                    self.clusters[comp].apply(lambda m: m['roll']).values,
                    2
                )[1]
        self._model_dict['spiral'] = {}
        for i in range(len(self.spiral_arms)):
            arm = self.spiral_arms[i]
            self._model_dict['spiral']['I.{}'.format(i)] = spirals[i][1]['I']
            self._model_dict['spiral']['spread.{}'.format(i)] = spirals[i][1]['spread']
            self._model_dict['spiral']['A.{}'.format(i)] = arm.A
            self._model_dict['spiral']['phi.{}'.format(i)] = arm.pa * arm.chirality
            self._model_dict['spiral']['t_min.{}'.format(i)] = arm.t_predict.min()
            self._model_dict['spiral']['t_max.{}'.format(i)] = arm.t_predict.max()
        unconstrained_errs = pd.concat((
            self.errors.xs('I', level=1, drop_level=False),
            self.errors.xs('n', level=1, drop_level=False),
            self.errors.xs('c', level=1, drop_level=False),
        ))
        self.errors[unconstrained_errs.index] = np.inf
        self.model = pd.DataFrame(self._model_dict).unstack().dropna()
        del self._model_dict
        # self.params = to_pandas(self.model).rename('model')

    def get_spirals(self):
        s = self.model['spiral']
        return np.array([
            inclined_log_spiral(
                s['t_min.{}'.format(i)], s['t_max.{}'.format(i)],
                s['A.{}'.format(i)], s['phi.{}'.format(i)],
                **self.model['disk']
            )
            for i in range(len(self.spiral_arms))
        ])
