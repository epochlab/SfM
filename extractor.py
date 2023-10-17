#!/usr/bin/env python3

import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

class FeatureExtraction():
    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher()
        self.prev = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

    def normalise(self, pts):
        return np.dot(self.Kinv, self.add_ones(pts).T).T[:, 0:2]

    def denormalise(self, pt):
        x = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        return int(round(x[0])), int(round(x[1]))

    def add_ones(self, x):
        """[[x, y]] -> [[x, y, 1]]"""
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

    def extract(self, x):    
        # Detection
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        feat = cv2.goodFeaturesToTrack(gray, maxCorners=3000, qualityLevel=0.01, minDistance=3) # Shi-Tomasi Corner Detector

        # Extraction
        kp = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feat]
        kp, des = self.orb.compute(gray, kp)

        # Matches
        ret = []
        if self.prev is not None:
            matches = self.bf.knnMatch(des, self.prev['des'], k=2)
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kp[m.queryIdx].pt
                    kp2 = self.prev['kp'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        # Filter
        if len(ret)>0:
            ret = np.array(ret)

            # Normalise Coords
            ret[:,0,:] = self.normalise(ret[:,0,:])
            ret[:,1,:] = self.normalise(ret[:,1,:])

            _, inliers = ransac((ret[:, 0], ret[:, 1]),
                                FundamentalMatrixTransform,
                                min_samples=8, residual_threshold=1, max_trials=100)

            ret=ret[inliers]

            # s,v,d = np.linalg.svd(model.params)
            # print(v)

        self.prev = {'kp': kp, 'des': des}
        return ret