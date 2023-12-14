import cv2
import numpy as np
import math
import csv
from scipy.spatial.distance import cdist, cosine, directed_hausdorff
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

class ShapeContext(object):

    def __init__(self, nbins_r=5, nbins_theta=12, r_inner=0.1250, r_outer=2.0):
        # number of radius zones
        self.nbins_r = nbins_r
        # number of angles zones
        self.nbins_theta = nbins_theta
        self.r_inner = r_inner
        self.r_outer = r_outer

    def _hungarian(self, cost_matrix):
        '''
        Similar points from 2 point sets based on cost matrices.
        Returns total modification cost, indices of matched points

        '''
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total = cost_matrix[row_ind, col_ind].sum()
        indexes = zip(row_ind.tolist(), col_ind.tolist())
        return total, indexes

    def get_points_from_img(self, image, threshold=50, simpleto=100, radius=2):
        ### creates a grid and gets points of the image on that grid
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dst = cv2.Canny(image, threshold, threshold * 3, 3)
        py, px = np.gradient(image)
        # px, py gradients maps shape can be smaller then input image shape
        points = [index for index, val in np.ndenumerate(dst)
                  if val == 255 and index[0] < py.shape[0] and index[1] < py.shape[1]]
        h, w = image.shape

        _radius = radius
        while len(points) > simpleto:
            newpoints = points
            xr = range(0, w, _radius)
            yr = range(0, h, _radius)
            for p in points:
                if p[0] not in yr and p[1] not in xr:
                    newpoints.remove(p)
                    if len(points) <= simpleto:
                        T = np.zeros((simpleto, 1))
                        for i, (y, x) in enumerate(points):
                            radians = math.atan2(py[y, x], px[y, x])
                            T[i] = radians + 2 * math.pi * (radians < 0)
                        return points, np.asmatrix(T)
            _radius += 1
        T = np.zeros((simpleto, 1))
        for i, (y, x) in enumerate(points):
            radians = math.atan2(py[y, x], px[y, x])
            T[i] = radians + 2 * math.pi * (radians < 0)
        return points, np.asmatrix(T)

    def _cost(self, hi, hj):
        cost = 0
        for k in xrange(self.nbins_theta * self.nbins_r):
            if (hi[k] + hj[k]):
                cost += ((hi[k] - hj[k])**2) / (hi[k] + hj[k])
        return cost * 0.5

    def cost_by_paper(self, P, Q, qlength=None):
        p, _ = P.shape
        p2, _ = Q.shape
        d = p2
        if qlength:
            d = qlength
        C = np.zeros((p, p2))
        for i in xrange(p):
            for j in xrange(p2):
                C[i, j] = self._cost(Q[j] / d, P[i] / p)
        return C

    def compute(self, points):
        ### Gives the shape context descriptor
        t_points = len(points)
        ### euclidian distance
        r_array = cdist(points, points)
        # getting two points with maximum distance to norm angle by them
        # this is needed for rotation invariant feature
        am = r_array.argmax()
        max_points = [am / t_points, am % t_points]
        # normalizing
        r_array_n = r_array / r_array.mean()
        # create log space
        r_bin_edges = np.logspace(np.log10(self.r_inner), np.log10(self.r_outer), self.nbins_r)
        r_array_q = np.zeros((t_points, t_points), dtype=int)
        for m in xrange(self.nbins_r):
            r_array_q += (r_array_n < r_bin_edges[m])
        fz = r_array_q > 0
        # getting angles in radians
        theta_array = cdist(points, points, lambda u, v: math.atan2((v[1] - u[1]), (v[0] - u[0])))
        norm_angle = theta_array[max_points[0], max_points[1]]
        # making angles matrix rotation invariant
        theta_array = (theta_array - norm_angle * (np.ones((t_points, t_points)) - np.identity(t_points)))
        # removing all very small values because of float operation
        theta_array[np.abs(theta_array) < 1e-7] = 0
        # 2Pi shifted because we need angels in [0,2Pi]
        theta_array_2 = theta_array + 2 * math.pi * (theta_array < 0)
        # Simple Quantization
        theta_array_q = (1 + np.floor(theta_array_2 / (2 * math.pi / self.nbins_theta))).astype(int)
        # building point descriptor based on angle and distance
        nbins = self.nbins_theta * self.nbins_r
        descriptor = np.zeros((t_points, nbins))
        for i in xrange(t_points):
            sn = np.zeros((self.nbins_r, self.nbins_theta))
            for j in xrange(t_points):
                if (fz[i, j]):
                    sn[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
            descriptor[i] = sn.reshape(nbins)
        return descriptor

    def cosine_diff(self, P, Q):
        P = P.flatten()
        Q = Q.flatten()
        assert len(P) == len(Q), 'keep no of descriptors same'
        return cosine(P, Q)

    def diff(self, P, Q, qlength=None):
        result = None
        C = self.cost_by_paper(P, Q, qlength)
        result = self._hungarian(C)
        return result

    def hausdorff(self, P, Q):
    	### Gives the directed Hausdorff distance between 2 point sets (2 arrays)
        return directed_hausdorff(P, Q)[0]

def plot(img, img2):
    sc = ShapeContext()
    sampls = 100
    points1, t1 = sc.get_points_from_img(img, simpleto=sampls)
    points2, t2 = sc.get_points_from_img(img2, simpleto=sampls)
    points2 = (np.array(points2) + 30).tolist()
    P = sc.compute(points1)
    x1 = [p[1] for p in points1]
    y1 = [p[0] for p in points1]
    Q = sc.compute(points2)
    x2 = [p[1] for p in points2]
    y2 = [p[0] for p in points2]
    # standard_cost, indexes = sc.diff(P, Q)
    # lines = []
    # for p, q in indexes:
    #     lines.append(((points1[p][1], points1[p][0]), (points2[q][1], points2[q][0])))
    # ax = plt.subplot(121)
    # plt.gca().invert_yaxis()
    # plt.plot(x1, y1, 'go', x2, y2, 'ro')
    # ax = plt.subplot(122)
    # plt.gca().invert_yaxis()
    # plt.plot(x1, y1, 'go', x2, y2, 'ro')
    # for p1, p2 in lines:
    #     plt.gca().invert_yaxis()
        # plt.plot((p1[0], p2[0]), (p1[1], p2[1]), 'k-')
    # plt.show()
    # plt.close()
    hsdrf = sc.hausdorff(P, Q)
    # print "Cosine diff:", cosine(P.flatten(), Q.flatten())
    # print "Standard diff:", standard_cost
    # print "Directed Hausdorff distance:", hsdrf
    return hsdrf

def match(image_path):
    test_image = cv2.imread(image_path, 0)
    distances = []
    for objid in range(20):
        for poseid in range(72):
            other_image = cv2.imread('./coil-20-proc/obj' + str(objid + 1) + '__' + str(poseid) + '.png', 0)
            hd = plot(test_image, other_image)
            distances.append(hd)
    minarg = np.argmin(np.array(distances))
    if ((minarg + 1) % 72) == 0:
        opid = (minarg + 1) / 72
        psid = 71
    else:
        opid = (minarg + 1) / 72
        opid = opid + 1
        psid = ((minarg + 1) % 72) - 1
    oplabel = 'obj' + str(opid) + '__' + str(psid)
    print 'Object in the image is', oplabel

test_image = './test_image_coil.png'
match(test_image)