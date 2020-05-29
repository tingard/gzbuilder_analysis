import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import GroupKFold
from gzbuilder_analysis.aggregation.spirals import \
    cleaning, deprojecting, fitting, utils, get_sample_weight


class Arm():
    def __init__(self, parent_pipeline, arms, clean_points=True,
                 weight_points=True):
        self.__parent_pipeline = parent_pipeline
        self.arms = np.asarray(utils.equalize_arm_length(arms))
        self.phi = parent_pipeline.phi
        self.ba = parent_pipeline.ba
        self.should_weight_points = weight_points
        self.image_size = parent_pipeline.image_size
        self.centre_pos = parent_pipeline.centre_pos
        self.FLAGGED_AS_BAD = False

        self.bar_length = parent_pipeline.bar_length / self.image_size[0]
        self.logsp_model = fitting.get_log_spiral_pipeline()

        self.coords, self.groups_all = cleaning.get_grouped_data(arms)
        if clean_points:
            self.outlier_mask = cleaning.clean_arms_xy(
                self.coords,
                self.groups_all,
            )
        else:
            self.outlier_mask = np.ones(len(self.coords), dtype=bool)
        self.did_clean = clean_points
        self.__get_deprojected_coordinates()
        self.__get_point_weights()
        self.__fit_logsp()

    def __get_deprojected_coordinates(self):
        self.deprojected_coords = deprojecting.deproject_arm(
            (self.coords - self.centre_pos) / self.image_size,
            angle=self.phi, ba=self.ba,
        )
        self.R_all, self.t_all = utils.r_theta_from_xy(
            *self.deprojected_coords.T
        )
        self.t_all_unwrapped = fitting.unwrap(self.t_all, self.R_all,
                                              self.groups_all)
        # only apply masking after unwrapping
        self.groups = self.groups_all[self.outlier_mask]
        self.R = self.R_all[self.outlier_mask]
        self.t = self.t_all_unwrapped[self.outlier_mask]

    def __get_point_weights(self):
        if self.should_weight_points:
            self.point_weights = utils.get_sample_weight(
                self.R, self.groups, self.bar_length
            )
        else:
            self.point_weights = np.ones_like(self.R)

    def __fit_logsp(self):
        # predict on the 1-99 percentile values of theta
        self.t_predict = np.linspace(
            np.percentile(self.t, 1),
            np.percentile(self.t, 99),
            500,
        )
        self.logsp_model.fit(self.t.reshape(-1, 1), self.R,
                             bayesianridge__sample_weight=self.point_weights)
        if self.logsp_model.score(self.t.reshape(-1, 1), self.R,) < 0.2:
            self.FLAGGED_AS_BAD = True

        R_predict = self.logsp_model.predict(self.t_predict.reshape(-1, 1))
        bar_mask = R_predict > self.bar_length
        t_predict = self.t_predict[bar_mask]
        R_predict = R_predict[bar_mask]
        self.polar_logsp = np.array((t_predict, R_predict))

        x, y = utils.xy_from_r_theta(R_predict, t_predict)
        self.log_spiral = np.stack((x, y), axis=1) * self.image_size + self.centre_pos
        self.reprojected_log_spiral = deprojecting.reproject_arm(
            arm=np.stack((x, y), axis=1),
            angle=self.phi,
            ba=self.ba
        ) * self.image_size + self.centre_pos
        self.length = np.sqrt(
            np.sum(
                np.diff(self.reprojected_log_spiral, axis=0)**2,
                axis=1
            ),
        ).sum()
        br = self.logsp_model.named_steps['bayesianridge'].regressor_
        self.coef = br.coef_
        self.sigma = br.sigma_
        self.A = np.exp(self.coef[0]) * self.image_size[0]
        self.pa, self.sigma_pa, self.chirality = utils.get_pitch_angle(
            self.coef[1],
            self.sigma[1, 1]
        )

    def modify_disk(self, centre=None, phi=None, ba=None):
        if phi is not None:
            self.phi = phi
        if ba is not None:
            self.ba = ba
        if centre is not None:
            delta = np.asarray(centre, dtype=float) - self.centre_pos
            self.centre_pos = np.asarray(centre, dtype=float)
            # self.arms -= delta
            # self.coords -= delta
        self.__get_deprojected_coordinates()
        self.__get_point_weights()
        self.__fit_logsp()
        return self.reprojected_log_spiral

    def get_error(self):
        foo = []
        for i in np.unique(self.groups):
            t = self.t[self.groups == i]
            R = self.R[self.groups == i]
            if len(t) < 10:
                continue
            pw = self.point_weights[self.groups == i]
            self.logsp_model.fit(np.expand_dims(t, 1), R,
                                 bayesianridge__sample_weight=pw)
            br = self.logsp_model.named_steps['bayesianridge'].regressor_
            coef = br.coef_
            sigma = br.sigma_
            pa, sigma_pa, chirality = utils.get_pitch_angle(
              coef[0],
              sigma[0, 0]
            )
            foo.append((pa, sigma_pa))
        foo = np.array(foo)

        return np.sqrt(np.sum(foo[:, 1]**2))

    def get_parent(self):
        return self.__parent_pipeline

    def fit_polynomials(self, min_degree=1, max_degree=5, n_splits=5,
                        score=median_absolute_error, lower_better=True):
        gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(self.groups))))
        models = {}
        scores = {}
        for degree in range(min_degree, max_degree):
            poly_model = fitting.get_polynomial_pipeline(degree)
            s = fitting.weighted_group_cross_val(
                poly_model,
                self.t.reshape(-1, 1), self.R,
                cv=gkf,
                groups=self.groups,
                weights=self.point_weights,
                score=score,
                lower_better=lower_better
            )
            poly_model.fit(self.t.reshape(-1, 1), self.R)
            poly_r = poly_model.predict(self.t_predict.reshape(-1, 1))
            models['poly_spiral_{}'.format(degree)] = np.stack(
                (self.t_predict, poly_r),
                axis=1
            )
            scores['poly_spiral_{}'.format(degree)] = s
        s = fitting.weighted_group_cross_val(
            self.logsp_model,
            self.t.reshape(-1, 1), self.R,
            cv=gkf,
            groups=self.groups,
            weights=self.point_weights,
            score=score,
            lower_better=lower_better
        )
        models['log_spiral'] = np.stack((
            self.t_predict,
            self.logsp_model.predict(self.t_predict.reshape(-1, 1))
        ), axis=1)
        scores['log_spiral'] = s
        return models, scores

    def __add__(self, other):
        if not (
            self.phi == other.phi
            and self.ba == other.ba
            and np.all(self.image_size == other.image_size)
        ):
            raise ValueError(
                'Cannot concatenate two arms with different '
                'deprojection values'
            )
        grouped_drawn_arms = np.concatenate([self.arms, other.arms])
        return Arm(self.__parent_pipeline, grouped_drawn_arms, self.did_clean)

    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as f:
            res = pickle.load(f)
        if not isinstance(res, cls):
            del res
            return False
        return res

    def save(self, fname):
        if not len(fname.split('.')) > 1:
            fname += '.pickle'
        with open(fname, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def get_info(self):
        return pd.Series({
            'n_arms': len(self.arms),
            'n_points': len(self.coords),
            'length': self.length,
            'log spiral pa': self.pa,
            'log spiral pa err': self.sigma_pa,
        })

    def _repr_html_(self):
        return (
            '<svg viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg" '
            'style="width:200px; height:200px">'
            '<polyline points="{}" fill="none" stroke="black" /></svg>'
        ).format(
            *self.image_size / 10,
            ' '.join(
                ','.join(map(str, point.round(2) / 10))
                for point in self.reprojected_log_spiral
            )
        )
