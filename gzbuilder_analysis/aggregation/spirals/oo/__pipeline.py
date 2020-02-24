import numpy as np
from sklearn.cluster import DBSCAN
from gzbuilder_analysis.config import COMPONENT_CLUSTERING_PARAMS
from gzbuilder_analysis.aggregation.spirals import utils, metric
from .__arm import Arm

SPIRAL_CLUSTERING_PARAMS = COMPONENT_CLUSTERING_PARAMS['spiral']

class SpiralPipeline(object):
    def __init__(
        self, drawn_arms,
        centre_pos=None, phi=0.0, ba=1.0,  # parameters for deprojection
        bar_length=0, centre_threshold=2.5,  # parameters for central cropping
        image_size=(100, 100),
        eps=SPIRAL_CLUSTERING_PARAMS['eps'],
        min_samples=SPIRAL_CLUSTERING_PARAMS['min_samples'],
    ):
        self.drawn_arms = np.array(
            utils.split_arms_at_center(
                np.array(drawn_arms),
                image_size=image_size[0],
                threshold=centre_threshold,
            )
        )
        self.image_size = np.asarray(image_size, dtype=float)
        if centre_pos is None:
            self.centre_pos = self.image_size / 2
        else:
            self.centre_pos = np.asarray(centre_pos, dtype=float)
        self.phi = float(phi)
        self.ba = float(ba)
        self.bar_length = float(bar_length)
        self.scaled_arms = np.array([
            arm / image_size for arm in self.drawn_arms
        ])
        self.distances = metric.calculate_distance_matrix_parallel(
            self.scaled_arms
        )
        self.cluster_arms(eps=eps, min_samples=min_samples)

    def cluster_arms(
        self,
        eps=SPIRAL_CLUSTERING_PARAMS['eps'],
        min_samples=SPIRAL_CLUSTERING_PARAMS['min_samples']
    ):
        self.db = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='precomputed',
            n_jobs=-1,
            algorithm='brute',
        ).fit(self.distances)

    def __get_arm(self, arm_label, clean_points=True, weight_points=True):
        return Arm(
            self,
            self.drawn_arms[self.db.labels_ == arm_label],
            clean_points=clean_points,
            weight_points=weight_points
        )

    def __filter_arms(self, arms):
        return [arm for arm in arms if not arm.FLAGGED_AS_BAD]

    def get_arms(self, merge=True, *args, **kwargs):
        if merge:
            return self.__filter_arms(
                self.merge_arms(
                    self.get_arms(merge=False, *args, **kwargs)
                )
            )
        return self.__filter_arms([
            self.__get_arm(i, *args, **kwargs)
            for i in range(max(self.db.labels_) + 1)
        ])

    def merge_arms(self, arms, threshold=SPIRAL_CLUSTERING_PARAMS['merging_distance']):
        arms = np.array(arms)
        logsps = [arm.reprojected_log_spiral / self.image_size for arm in arms]
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
