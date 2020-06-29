# vim: expandtab:ts=4:sw=4
import numpy as np

fyj_num_nn = 2
def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    # print(a[0])
    # print('*************')
    # print(len(a),',128')
    # print(len(b),',128')
    # ff_file = open('E:/project/multi_track/deep_sort/tmp/02.txt','a')
    # ff_file.write('*******************\n')
    # ff_file.write(str(a)+'\n')
    # ff_file.write('-------------------\n')
    # ff_file.write(str(b)+'\n')
    



    
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    # print('----------->')
    # print(distances.min(axis=0))
    # print('-----------')
    # print(x)
    # distance_arr = distances.min(axis=0)
    # print(distance_arr)
    # print(len(distance_arr))
    # print(distances.size)
    # fff = open('E:/project/multi_track/deep_sort/tmp/output_02.txt','a')
    # fff.write('*******************\n')
    # fff.write(str(distances)+'\n')
    # fff.write(str(distances.min(axis=0))+'\n')
    # fff.write(str(np.argmin(distances,axis=0))+'\n')
    # print(distances)
    # print(distances.min(axis=0))

    '''
    # global fyj_num_nn
    # fyj_num_nn = fyj_num_nn + 1 

    # f_distances = open('E:/project/multi_track/deep_sort/tmp/distances/{:05d}.txt'.format(fyj_num_nn),'a')
    # f_distances.write(str(np.argmin(distances,axis=0))+'\n')

    # f_xfeature = open('E:/project/multi_track/deep_sort/tmp/x_features/{:05d}.txt'.format(fyj_num_nn),'a')
    # f_xfeature.write(str(x)+'\n')

    # f_yfeature = open('E:/project/multi_track/deep_sort/tmp/y_features/{:05d}.txt'.format(fyj_num_nn),'a')
    # f_yfeature.write(str(y)+'\n')
    '''
    
    # print('------->distance')
    # print(np.argmin(distances,axis=0))

    # for i in range(len(distance_arr)):
    #     print('~~~~~~~~~~~~~~~~~~~')
    #     for j in range(int(distances.size/len(distance_arr))):
    #         print(distances[j][i])
    # print('Dis len:{}'.format(len(distances)))
   
    # print(len(distance_arr))
    # for line in range(len(distance_arr)):
    #     print('-----')
    #     tmp_dis = distances[line]
    #     tmp_line = x[line]
    #     print('index {}'.format(np.argmin(distances)))
    #     print('data {}'.format(tmp_line(np.argmin(distances))))


    #     print('Match index:{}'.format(np.argmin(line)))
    # ff_file1 = open('E:/project/multi_track/deep_sort/tmp/temp_02.txt','a')
    # ff_file1.write('*******************\n')
    # ff_file1.write(str(x[np.argmin(distances)])+'\n')
    # ff_file1.write('-------------------\n')
    # ff_file1.write(str(y)+'\n')
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None):


        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold # default is 0.3
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}
        # print('--samples--')
        # print(self.samples)

    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        cost_matrix = np.zeros((len(targets), len(features)))

        # fyj_features = []
        # print(len(targets))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
            # print('^^^^^^^^^^^^^^^^^')
            # fyj_features=features
            
            # fyj_arr.append(temp.tolist())
        global fyj_num_nn
        fyj_num_nn = fyj_num_nn + 1
     

        # print('----nn_matching----{}'.format(fyj_num_nn))

        # print(fyj_features)
        return cost_matrix
