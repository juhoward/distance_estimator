import cv2 as cv
from statistics import median
from math import dist
import numpy as np
import os
import glob


class MonoCalibrator(object):
    '''
    objective of this object is to estimate the camera's focal length.
    The best method right now is the mono_calibrate, based on Zhang's
    checkerboard calibration and implemented in opencv.

    f : estimated focal length
    points3D : numpy array holding hypthetical points in 3D space
    points2D : numpy array holding detected checkerboard points
    checker_dims : the dimensions of the checkerboard calibration target
    grayframe : the last captured grayscale frame, stored to get re-projeciton error
    '''
    def __init__(self,
                 checker_dims = (5,8)):
        os.system("bash ./calibration/disable_autofocus.sh")
        self.f = 0
        # Vector for 3D points
        self.points3D= []
        # Vector for 2D points
        self.points2D = []
        # counter
        self.cnt = 0
        # checkerboard dimensions
        self.checker_dims = checker_dims
        # save grayscale frame
        self.grayFrame = None

    def show_iris_kpts(self, img):
        left_i = self.face.mesh[self.detector.LEFT_IRIS]
        right_i = self.face.mesh[self.detector.RIGHT_IRIS]
        for idx, pt in enumerate(left_i):
            cv.circle(img, pt, 1, (255,0,255), 1, cv.LINE_AA)
            cv.putText(img, str(self.detector.LEFT_IRIS[idx]), (pt[0],pt[1]-5),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv.LINE_AA)
        for idx, pt in enumerate(right_i):
            cv.circle(img, pt, 1, (255,0,255), 1, cv.LINE_AA)
            cv.putText(img, str(self.detector.RIGHT_IRIS[idx]), (pt[0],pt[1]-5),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv.LINE_AA)
        return img

    def get_f_length(self, w_pixels, w_mm):
        '''
        returns focal length based on triangle similarity.

        self.d_2_obj = distance to camera (mm)
        w_pixels = pixel width of detected object
        w_mm = real / assumed width of object (mm)
        '''
        return (self.d_2_obj * w_pixels) / w_mm
    
    def mono_calibration_data_reader(self, img_dir):
        '''
        reads previously collected calibration images.
        '''
        imgs = glob.glob('./calibration/mono_imgs/*.png')
        return (cv.cvtColor(cv.imread(im), cv.COLOR_BGR2GRAY) for im in imgs)

    def find_corners(self, img_reader, checker_dims=(5,8)):
        criteria = (cv.TERM_CRITERIA_EPS +
                    cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 3D points real world coordinates
        objectp3d = np.zeros((1, 
                            checker_dims[0] * checker_dims[1],
                            3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:checker_dims[0],
                                    0:checker_dims[1]].T.reshape(-1, 2)
        for img in img_reader:
            ret, corners = cv.findChessboardCorners(
                            img, checker_dims,
                            cv.CALIB_CB_ADAPTIVE_THRESH
                            + cv.CALIB_CB_FAST_CHECK +
                            cv.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                self.grayFrame = img
                self.points3D.append(objectp3d)
                corners2 = cv.cornerSubPix(
                            self.grayFrame, corners, (11, 11), (-1, -1), criteria)
                self.points2D.append(corners2)
    
    def get_mono_calibration_data(self, path, frame, checker_dims=(5,8), save=True):
        '''
        1. uses single camera to collect calibration images.
        2. if checkerboard of check_dims dimensions is detected, 2 and 3d points are collected.
        '''
        # stop the iteration when specified
        # accuracy, epsilon, is reached or
        # specified number of iterations are completed.
        criteria = (cv.TERM_CRITERIA_EPS +
                    cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 3D points real world coordinates
        objectp3d = np.zeros((1, 
                            checker_dims[0] * checker_dims[1],
                            3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:checker_dims[0],
                                    0:checker_dims[1]].T.reshape(-1, 2)
        self.grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv.findChessboardCorners(
                        self.grayFrame, checker_dims,
                        cv.CALIB_CB_ADAPTIVE_THRESH
                        + cv.CALIB_CB_FAST_CHECK +
                        cv.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            self.cnt += 1
            if save == True:
                fname = os.path.join(path, f'{self.cnt}.png')
                print(f'Saving image: {fname}')
                cv.imwrite(fname, frame)
            self.points3D.append(objectp3d)
            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv.cornerSubPix(
                self.grayFrame, corners, (11, 11), (-1, -1), criteria)
            self.points2D.append(corners2)
            # Draw and display the corners
            image = cv.drawChessboardCorners(frame,
                                            checker_dims,
                                            corners2, ret)
            return image

    def mono_calibrate(self):
        ''' 
        Perform camera calibration by
        passing the 3D points and the corresponding 
        pixel coordinates of the detected corners (points2D)
        '''
        ret, matrix, distortion, r_vecs, t_vecs = cv.calibrateCamera(
            self.points3D, self.points2D, self.grayFrame.shape[::-1], None, None)
        if ret < .3:
            print('Calibration successful.')
            print(f'Calibration error:\n{ret}')
            print(f"\nCamera matrix: \n{matrix}")
            return ret, matrix, distortion, r_vecs, t_vecs
        else:
            print('Calibration unsuccessful. Try taking more calibration pictures.')

if __name__ == '__main__':
    cameras = {'camL':0}

    # calibrator, where distance to camera (in.) is transformed (mm)
    monocal = MonoCalibrator()
    # updated face object with focal length and iris width (mm)
    face = monocal.stream(from_saved=True)
    print(f'Focal Length: {monocal.f}')