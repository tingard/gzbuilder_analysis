import json
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import GroupKFold
from gzbuilder_analysis.config import ARM_CLUSTERING_PARAMS
from gzbuilder_analysis.aggregation.spirals import equalize_arm_length
from gzbuilder_analysis.aggregation.spirals import split_arms_at_center
from gzbuilder_analysis.aggregation.spirals import xy_from_r_theta, r_theta_from_xy
from gzbuilder_analysis.aggregation.spirals import metric, cleaning
from gzbuilder_analysis.aggregation.spirals import deprojecting, fitting
from gzbuilder_analysis.aggregation.spirals import get_pitch_angle, get_sample_weight


class Arm():
    def __init__(self, parent_pipeline, arms, clean_points=True):
        self.__parent_pipeline = parent_pipeline
        self.arms = np.array(equalize_arm_length(arms))
        self.phi = parent_pipeline.phi
        self.ba = parent_pipeline.ba
        self.image_size = parent_pipeline.image_size
        self.FLAGGED_AS_BAD = False

        bar_length = parent_pipeline.bar_length / self.image_size

        self.logsp_model = fitting.get_log_spiral_pipeline()

        self.coords, self.groups_all = cleaning.get_grouped_data(arms)
        self.deprojected_coords = deprojecting.deproject_arm(
            self.coords / self.image_size - 0.5,
            angle=self.phi, ba=self.ba,
        )
        self.R_all, self.t_all = r_theta_from_xy(*self.deprojected_coords.T)
        self.t_all_unwrapped = fitting.unwrap(self.t_all, self.R_all,
                                              self.groups_all)
        if clean_points:
            self.outlier_mask = cleaning.clean_arms_xy(
                self.coords,
                self.groups_all,
            )
        else:
            self.outlier_mask = np.ones(self.R_all.shape[0], dtype=bool)
        self.did_clean = clean_points
        self.groups = self.groups_all[self.outlier_mask]
        self.R = self.R_all[self.outlier_mask]
        self.t = self.t_all_unwrapped[self.outlier_mask]
        self.point_weights = get_sample_weight(self.R, self.groups, bar_length)
        self.logsp_model.fit(self.t.reshape(-1, 1), self.R,
                             bayesianridge__sample_weight=self.point_weights)
        if self.logsp_model.score(self.t.reshape(-1, 1), self.R,) < 0.2:
            self.FLAGGED_AS_BAD = True

        self.t_predict = np.linspace(min(self.t), max(self.t), 500)
        R_predict = self.logsp_model.predict(self.t_predict.reshape(-1, 1))
        t_predict = self.t_predict[R_predict > bar_length]
        R_predict = R_predict[R_predict > bar_length]

        self.polar_logsp = np.array((t_predict, R_predict))

        x, y = xy_from_r_theta(R_predict, t_predict)

        self.length = np.sqrt(
            np.diff(x)**2 + np.diff(y)**2
        ).sum() * self.image_size

        self.log_spiral = (np.stack((x, y), axis=1) + 0.5) * self.image_size

        self.reprojected_log_spiral = (deprojecting.reproject_arm(
            arm=np.stack((x, y), axis=1),
            angle=self.phi,
            ba=self.ba
        ) + 0.5) * self.image_size
        br = self.logsp_model.named_steps['bayesianridge'].regressor_
        self.coef = br.coef_
        self.sigma = br.sigma_
        self.pa, self.sigma_pa, self.chirality = get_pitch_angle(
            self.coef[0],
            self.sigma[0, 0]
        )

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
            pa, sigma_pa, chirality = get_pitch_angle(
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
            self.image_size / 10,
            self.image_size / 10,
            ' '.join(
                ','.join(map(str, point.round(2) / 10))
                for point in self.reprojected_log_spiral
            )
        )


class Pipeline():
    def __init__(self, drawn_arms, phi=0.0, ba=1.0, distances=None,
                 bar_length=10, centre_size=20, image_size=512,
                 clustering_kwargs={}, parallel=False):
        self.drawn_arms = np.array(
            split_arms_at_center(
                np.array(drawn_arms),
                image_size=image_size,
                threshold=centre_size,
            )
        )
        self.image_size = float(image_size)
        self.phi = float(phi)
        self.ba = float(ba)
        self.bar_length = bar_length
        if distances is None:
            cdm = (metric.calculate_distance_matrix_parallel if parallel
                   else metric.calculate_distance_matrix)
            if parallel:
                self.distances = cdm(self.drawn_arms)
            else:
                self.distances = cdm(self.drawn_arms)
        else:
            self.distances = distances

        self.db = DBSCAN(
            eps=clustering_kwargs.setdefault(
                'eps',
                ARM_CLUSTERING_PARAMS['eps']
            ),
            min_samples=clustering_kwargs.get(
                'min_samples',
                ARM_CLUSTERING_PARAMS['min_samples']
            ),
            metric='precomputed',
            n_jobs=-1,
            algorithm='brute',
        ).fit(self.distances)

    def get_arm(self, arm_label, clean_points=True):
        arms_in_cluster = self.drawn_arms[self.db.labels_ == arm_label]
        return Arm(self, arms_in_cluster, clean_points=clean_points)

    def filter_arms(self, arms):
        return [arm for arm in arms if not arm.FLAGGED_AS_BAD]

    def get_arms(self, merge=True, *args, **kwargs):
        if merge:
            return self.filter_arms(
                self.merge_arms(
                    self.get_arms(merge=False, *args, **kwargs)
                )
            )
        return self.filter_arms([
            self.get_arm(i, *args, **kwargs)
            for i in range(max(self.db.labels_) + 1)
        ])

    def merge_arms(self, arms, threshold=500):
        arms = np.array(arms)
        logsps = [arm.reprojected_log_spiral for arm in arms]
        pairs = []
        for i in range(len(logsps)):
            for j in range(i+1, len(logsps)):
                a, b = logsps[i], logsps[j]
                min_dist = min(
                    np.sum(metric._npsdtp_vfunc(a, b)) / len(a),
                    np.sum(metric._npsdtp_vfunc(b, a)) / len(b),
                )
                if min_dist <= threshold:
                    pairs.append([i, j])
        pairs = np.array(pairs)
        # we have a list of duplicate pairs, now check if we should merge more
        # than two arms at a time
        groups = []
        for i, pair in enumerate(pairs):
            if not np.any(np.isin(pair, groups)):
                groups.append(pair)
                continue
            for i in range(len(groups)):
                if np.any(np.isin(pair, groups[i])):
                    groups[i] = np.unique(np.concatenate((groups[i], pair)))
        groups += [[i] for i in range(len(arms)) if i not in pairs]
        merged_arms = []
        for group in groups:
            if len(group) == 1:
                merged_arms.append(arms[group][0])
            else:
                grouped_drawn_arms = sum(
                    (list(a.arms) for a in arms[group]),
                    []
                )
                new_arm = Arm(
                    self,
                    grouped_drawn_arms,
                    clean_points=any(a.did_clean for a in arms[group])
                )
                merged_arms.append(new_arm)
        return np.array(merged_arms)

    def get_pitch_angle(self, arms=None):
        if arms is None:
            arms = self.get_arms()
        if len(arms) == 0:
            return np.nan, np.nan
        pa = np.zeros(len(arms))
        sigma_pa = np.zeros(pa.shape)
        length = np.zeros(pa.shape)
        for i, arm in enumerate(arms):
            pa[i] = arm.pa
            length[i] = arm.length
            sigma_pa[i] = arm.sigma_pa
        combined_pa = (pa * length).sum() / length.sum()
        combined_sigma_pa = (
            np.sqrt((length**2 * sigma_pa**2).sum())
            / length.sum()
        )
        return combined_pa, combined_sigma_pa

    @classmethod
    def load(cls, fname, parallel=True):
        with open(fname) as f:
            obj = json.load(f)
        return cls(
            list(map(np.array, obj.get('drawn_arms', []))),
            phi=obj.get('phi', 0.0),
            ba=obj.get('ba', 1.0),
            distances=np.array(obj.get('distances', None)),
            bar_length=obj.get('bar_length', 10),
            image_size=obj.get('image_size', 512),
            parallel=parallel,
        )

    def save(self, fname):
        if not len(fname.split('.')) > 1:
            fname += '.json'
        with open(fname, 'w') as f:
            json.dump({
                'drawn_arms': list(map(np.ndarray.tolist, self.drawn_arms)),
                'phi': self.phi,
                'ba': self.ba,
                'distances': self.distances.tolist(),
                'bar_length': self.bar_length,
                'image_size': self.image_size,
            }, f)
