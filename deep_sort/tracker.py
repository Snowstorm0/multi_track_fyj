# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track

rect_set = [[[0 for i in range(4)] for j in range(2)] for k in range(20)]  # 连续2帧，3个轨迹，(xmin,ymin,xmax,ymax)
last_rect_set = [[0 for io in range(4)]  for ko in range(20)]

fyj_num_tracker = 0
class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=10000, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.three_D_arr = [[[0 for i in range(128)] for j in range(3)] for k in range(12)]
        # self.three_D_score = [[0 for i in range(3)] for j in range(12)]

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def fyj_cosine_distance(self,a, b):
        a = np.asarray(a) / np.linalg.norm(a, ord=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, ord=1, keepdims=True)
        return  np.dot(a, b)
    def compute_iou(self,rect1,rect2):
        S_rect1 = (rect1[3] - rect1[1]) * (rect1[2] - rect1[0])
        S_rect2 = (rect2[3] - rect2[1]) * (rect2[2] - rect2[0])
        sum_area = S_rect1 +S_rect2
        left_line = max(rect1[0], rect2[0])
        right_line = min(rect1[2], rect2[2])
        top_line = max(rect1[1], rect2[1])
        bottom_line = min(rect1[3], rect2[3])
        if(left_line >= right_line or top_line >= bottom_line):
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return intersect / (sum_area - intersect)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            # fyj_score = []
            # for id_chennel in range(12):
            #     fyj_score.append((self.fyj_cosine_distance(self.three_D_arr[id_chennel],(detections[detection_idx].feature))).tolist())

            # self.three_D_arr[track_idx].pop(0)
            # print('---kkkkkk---')
            # print(self.tracks)
            # print(fyj_score)
            # self.three_D_arr[track_idx].append((detections[detection_idx].feature).tolist())

        # print(self.three_D_arr[:2])



        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # print('---active_targets---')
        # print(active_targets)

        features, targets = [], []
        for track in self.tracks:
            # print('tracks')
            # print('length is --{}'.format(len(self.tracks)))

            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)




    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)


            # print('--tracker--')
            # print(cost_matrix.shape)

            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # save last update rect
        for ii in range(len(last_rect_set)):
            last_rect_set[ii] = rect_set[ii][-1]

        # Split track set into confirmed and unconfirmed tracks.


        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        print('detection length--{}'.format(len(detections)))

        print('---track id---')
        for track in self.tracks:

            print(track.track_id)
        # [(0,0),(1,1)]
        print('---before update matches---')
        print(matches)
        # print(matches_b)
        if(len(matches)!=0):
            for mm in range(len(matches)):
                gt_index = self.tracks[matches[mm][0]].track_id-1
                rect_set[gt_index].pop(0)
                # print(self.tracks[matches[mm][0]].track_id-1)
                rect_set[gt_index].append((detections[matches[mm][1]].to_tlbr()).tolist())

        print('unmatched_detections--{}'.format(unmatched_detections))
        for unm_de in range(len(unmatched_detections)):
            unm_d = unmatched_detections[unm_de]
            un_de_bbox = detections[unm_d].to_tlbr().tolist()
            for mm in range(len(rect_set)):
                if(rect_set[mm][0]!=[0,0,0,0]):
                    for mmm in rect_set[mm]:
                        iou_de_ma = self.compute_iou(mmm,un_de_bbox)
                        # print('------->IOUUUU--{}'.format(iou_de_ma))
                        if(iou_de_ma>0.45):

                            print('this is belongs to track--{}'.format(mm+1)) 
                            # print('unmatched id-{}'.format(unm_de))
                            if(last_rect_set[mm]!=rect_set[mm][-1]): # 说明有更新，只不过有重复的检测结果
                                unmatched_detections.pop(unm_de)
                            if(last_rect_set[mm]==rect_set[mm][-1]): # 说明没有更新，只是重新出现
                                unmatched_detections.pop(unm_de)
                                
                                matches = matches + [(mm,unm_d)]
                            break

        print('---after update matches---')
        print(matches)
        # print('--last--')
        # print(last_rect_set)
        # print('--latest--')
        # print(rect_set)



        # if(len(matches)!=0):
        #     global fyj_num_tracker
        #     fyj_num_tracker = fyj_num_tracker + 1
            # f_matches = open('E:/project/multi_track/deep_sort/tmp/02/matches/{:05d}.txt'.format(fyj_num_tracker),'w')
            # f_matches.write(str(matches)+'\n')
            # print('----matches--->{}'.format(fyj_num_tracker))
            # print(matches)



        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        print('*******************')
        print(self._next_id)
        self._next_id += 1
