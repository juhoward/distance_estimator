from math import sqrt, dist
from statistics import median
import numpy as np

class FaceDet(object):
    '''
    Class representing a detected face & holds the following data:
    --mesh
    --head measurements
    --distance calculations

    Uses the assumed iris diameter of 11.7mm unless updated.
    TODO: 
    merge mesh indices from detector into face object to centralize face
    related data.
    transfer distance estimation to stereo vid stream script
    transfer error and history storage to results objects
    '''
    def __init__(self):
        # credit card width (mm)
        self.w_card = 85.6
        # mean human iris diameter (mm)
        self.w_iris = 11.7
        # iris data
        self.l_iris = {'center': None, 'radius': None}
        self.r_iris = {'center': None, 'radius': None}
        self.i_diameter = 0
        # mediapipe face mesh
        self.mesh = None
        # mesh indices for iris points
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        # mediapipe head pts
        self.head_pts = None
        # head width (mm) based on iris diameter (mm)
        self.head_w = 0
        # head pixel width
        self.head_pixw = 0
        # holds head measurements
        self.head_measurements = []
        # subject-to-camera distance using credit card(in)
        self.s2c_d = 0
        self.s2c_ds = []
        # subject-to-camera distance using f_iris (in)
        self.s2c_d_i = 0

        # grouund truth s2c distances (stereo-based)
        self.gt_s2c = 0
        self.gt_s2cs = []
        
        # average relative inverse depth (iris, head)
        self.ri_depth = 0
        self.ri_depths = []
        # converted absolute depth
        self.abs_depth = 0
        self.abs_depths = []
        #  errors
        self.error = 0
        self.errors = []

    def s2c_dist(self, f, w_object, w_pix, inches=True):
        '''
        returns the subject-to-camera distance in mm using triangle similarity.
        f : focal length in pixels
        w_object : known width of object in mm
        w_pix : object width in pixels
        '''
        # subject to camera distaince (mm)
        s2c_d = (f * w_object) / w_pix

        # transform mm to cm
        s2c_d /= 10
        # log metric distance (cm) for parameter estimation
        self.s2c_ds.append(s2c_d)
        if inches == True:
            # return distance in inches
            s2c_d = s2c_d / 2.54
            s2c_d_i = s2c_d_i / 2.54
        else:
            # return distance in ft
            s2c_d = self.cm_to_ft(s2c_d)
        # keep state for reporting
        self.s2c_d = s2c_d
        self.s2c_d_i = s2c_d_i

    def get_iris_diameter(self):
        '''
        returns the median iris diameter (pixels) using the 8 iris keypoints
        in the face mesh.
        '''
        # 4 iris points per eye
        kpts = [self.mesh[self.LEFT_IRIS],
                self.mesh[self.RIGHT_IRIS]]
        measurements = []
        for pts in kpts:
            # 2 euclidean distances per eye
            diameter1 = dist(pts[0], pts[2])
            diameter2 = dist(pts[1], pts[3])
            measurements.append(diameter1)
            measurements.append(diameter2)
        # returns median of the 4 diameters
        self.i_diameter = median(measurements)
        return self.i_diameter

    def get_w_iris(self, w_card, card_w_pix, card=True):
        '''
        uses width of the credit card to estimate the corneal diameter
        of detected irises.
        assumes that face and card were coplanar when detected.
        See calibrator object for self.d_2_obj value
        
        w_card: width of credit card (mm)
        card_w_pix: pixel width of credit card
        dmtr: pixel width of iris

        TODO: check this for accuracy
        '''
        # return pixel width of iris
        dmtr = self.get_iris_diameter()
        # return real iris diameter
        return (dmtr * w_card) / card_w_pix
    
    def update_iris_width(self, w_card, card_w_pix):
        '''
        changes face object's iris diameter (mm)
        '''
        self.w_iris = self.get_w_iris(w_card, card_w_pix)

    def get_headw(self, p1, p2, logging=True):
        '''
        takes cheekbone points from facemesh &
        returns the width (mm) of the head based on the iris detection.
        appends the head width in a list for later use. 
        p1 & p2 are tuples (x, y) representing detected head points.
        '''
        # head width in pixels
        self.head_pixw = dist(p1, p2)
        # horizontal distance in mm/pixel units : iris plane
        if self.i_diameter is not None:
            head_w = (self.head_pixw * self.w_iris) / self.i_diameter
            if logging:
                self.head_measurements.append(head_w)
                self.head_w = median(self.head_measurements)

    def get_depth(self, img):
        '''
        returns the average relative inverse depth of 2 depth pixels.
        '''
        if self.mesh is not None:
            # if face detected, use iris location depth
            l_ctr = list(map(lambda x: int(x), self.l_iris['center']))
            r_ctr = list(map(lambda x: int(x), self.r_iris['center']))
            # correction for out of image points
            for idx, (i,j) in enumerate(zip(l_ctr, r_ctr)):
                if idx == 0:
                    l_ctr[idx] = min(img.shape[0]-1, i)
                    r_ctr[idx] = min(img.shape[0]-1, j)
                else:
                    l_ctr[idx] = min(img.shape[1]-1, i)
                    l_ctr[idx] = min(img.shape[1]-1, j)                  
            left = img[l_ctr[0],l_ctr[1]]
            right = img[r_ctr[0], r_ctr[1]]
            ri_depth = (left + right) / 2
            self.ri_depth = ri_depth
            self.ri_depths.append(ri_depth)
        elif self.head_pts is not None:
            # use head pts from body model
            # correction for out of image points
            l_ctr = list(map(lambda x: int(x), self.head_pts[0]))
            r_ctr = list(map(lambda x: int(x), self.head_pts[1]))
            for idx, (i,j) in enumerate(zip(l_ctr, r_ctr)):
                if idx == 0:
                    l_ctr[idx] = min(img.shape[0]-1, i)
                    r_ctr[idx] = min(img.shape[0]-1, j)
                else:
                    l_ctr[idx] = min(img.shape[1]-1, i)
                    l_ctr[idx] = min(img.shape[1]-1, j) 
            left = img[l_ctr[0],l_ctr[1]]
            right = img[r_ctr[0], r_ctr[1]]
            ri_depth = (left + right) / 2
            self.ri_depth = ri_depth
            self.ri_depths.append(ri_depth)
        else:
            print("no object detected.")
            self.ri_depth = 0

    def rel2abs_2(self, pred_depths, gt_depths):
        '''
        given dataset of relative inverse depths and gt_depths (cm),
        finds a linear relationship in form pred = mx + b
        returns absolute depth (cm).
        '''
        # invert gt
        gt = list(map(lambda x: 1/x, gt_depths))
        # align prediction based on least squares estimates
        A = np.vstack([gt, np.ones(len(gt))]).T
        self.m, self.b = np.linalg.lstsq(A, pred_depths, rcond=None)[0]
        # transform to ft
        self.abs_depth = self.cm_to_ft(self.ri_depth * self.m + self.b)
    
    def rel2abs(self):
        '''
        a simple linear transformation.
        only works on distances less than 36 in.
        '''
        abs_depth = 1000 / self.ri_depth
        self.abs_depth = abs_depth
        self.abs_depths.append(abs_depth)
    
    def rmse(self, feet=False):
        '''
        returns rmse of converted abs depths and s2c distances.
        '''
        if feet:
            errors = list(map(lambda x: (self.cm_to_ft(x[0]) - x[1])**2, zip(self.s2c_ds, self.abs_depths)))
        else:
            # error in inches
            errors = list(map(lambda x: ((x[0] / 2.54) - x[1])**2, zip(self.s2c_ds, self.abs_depths)))
        return sqrt((sum(errors)/ len(errors)))

    def mae(self, feet=False):
        '''
        returns mean absolute error of converted abs depth and s2c distances
        '''
        if feet:
            errors = list(map(lambda x: abs(self.cm_to_ft(x[0]) - x[1]), zip(self.s2c_ds, self.abs_depths)))
        else:
            # inches
            errors = list(map(lambda x: abs((x[0] / 2.54) - x[1]), zip(self.s2c_ds, self.abs_depths)))
        return sum(errors) / len(errors)

    def mm2cm(self, dist):
        return dist/10

    def cm_to_ft(self, dist):
        return round(dist/(2.54*12), 2)

    def in_to_mm(self, dist):
        return round(dist * 2.54 * 10, 2)

    def diameter(self, radius):
        return int(radius * 2)

    def dist_euclid(self, pt1:tuple, pt2:tuple):
        return sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)