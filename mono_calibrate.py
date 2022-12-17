import cv2 as cv
from face import FaceDet
from detectors import PersonDetector
from statistics import median
from math import dist
import numpy as np
import os
import glob


class MonoCalibrator(object):
    '''
    objective of this object is to estimate the camera's focal length.
    The best method right now is the mono_calibrate, based on Zhang's
    checkerboard calibration and implemented in opencv
    The other two methods use the triangle similarity to estimate f
    based on the assumption of a standard sized credit card and that
    the detected iris is of standard width.
    d_2_obj : the distance from which the iris and card are detected
    face : face object to hold mesh data and measure distances
    detector : person detector that called up iris and body pose models
    camera_ids : the name and bus id of cameras
    points : two points representing the width of a card at the d_2_obj distance
    checker_dims : the dimensions of the checkerboard calibration target
    '''
    def __init__(self,
                 checker_dims = (5,8)):
        os.system("bash ./calibration/disable_autofocus.sh")
        self.f_monocal = 0
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


    def stream(self, from_saved=False):      
        f_lengths = []
        w_irises = []
        if from_saved:
            print(f'reading saved calibration images from : ./calibration/mono_imgs/')
            # reused saved claibration images
            img_reader = self.mono_calibration_data_reader('./calibration/mono_imgs/')
            self.find_corners(img_reader, (5,8))
            camera_intrinsics = self.mono_calibrate()
            # get mean of focal lengths in x and y dimensions
            f = (camera_intrinsics[1][0][0] + camera_intrinsics[1][1][1]) / 2
            self.f_monocal = f
            print(f'Calibration complete. focal length: {self.f_monocal}')
            print('press and hold spacebar when 12 in. from camera to calibrate iris...')
            print('press "n" to end calibration...')
        else:
            print('press and hold "c" when target is near camera ...')
        while True:
            for name, cam in self.cameras.items():
                ok, frame = cam.read()
                if ok:
                    # for finding card points at a new distance by hand
                    # cv.circle(frame, (257,240), 1, (255,0,255), 2, cv.LINE_AA)
                    # cv.circle(frame, (402,240), 1, (255,0,255), 2, cv.LINE_AA)
                    cv.imshow('calibration', frame)
            if cv.waitKey(2) & 0xff == ord('c'):
                # collect new calibration images
                self.get_mono_calibration_data(frame)
                if len(self.points2D) > 9:
                    # reset img count
                    self.cnt = 0
                    # intrinsic matrix is at index 1 of this list 
                    camera_intrinsics = self.mono_calibrate()
                    # get mean of focal lengths in x and y dimensions
                    f = (camera_intrinsics[1][0][0] + camera_intrinsics[1][1][1]) / 2
                    self.f_monocal = f
                    cv.destroyAllWindows()
                    # cue to user to move on to next procedure
                    print('Estimating focal length based on iris detector')
                    print('press and hold spacebar when 12 in. from camera ...')

            elif cv.waitKey(3) & 0xff == ord(' '):
                # estimate focal length
                output = self.detect(self.face, self.detector, frame)
                if output[0] == True:
                    self.cnt +=1
                    f, w_iris = output[1:]
                    f_lengths.append(f)
                    w_irises.append(w_iris)
                    m = f'Captured {self.cnt} out of 40 images.'
                    m2 = f'Iris diameter - {round(w_iris, 2)}'
                    cv.putText(frame, m, (50,50),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)
                    cv.putText(frame, m2, (50,100),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)
                    self.detector.visualize(frame)
                    # frame = self.show_iris_kpts(frame)
                    cv.imshow('iris capture from 12in.', frame)
                if self.cnt > 29:
                    # update iris-based focal length 
                    self.f_iris = median(f_lengths)
                    # update width of the iris
                    self.face.w_iris = median(w_irises)
                    print(f'credit card-based iris diameter: {median(w_irises)}')
                    for name in self.cameras:
                        self.cameras[name].release()
                    cv.destroyAllWindows()
                    return self.f_monocal, self.face.w_iris

    def detect(self, face, detector, frame):
        face.mesh = None        
        detector.findIris(frame)
        output = []
        if not face.mesh is None:
            output.append(True)
            # a tuple (left eye, right eye)
            face.xvals = self.xvals(face, detector)
            # returns median pixel width of iris
            w_iris_pix = face.get_iris_diameter()
            # transforms to real width and updates face mesh
            face.update_iris_width(self.w_card, self.card_w_pix)
            # focal length when iris diameter assumed to be 11.7mm
            output.append(self.get_f_length(w_iris_pix, 11.7))
            output.append(face.w_iris)
            return output
        else:
            output.append(False)
            return output

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

    def xvals(self, face, iris=True):
        '''
        collect x values from keypoint detections. x values are used for disparity
        '''
        if iris:
            xvals_left_i = list(map(lambda x: x[0], face.mesh[face.LEFT_IRIS]))
            xvals_right_i = list(map(lambda x: x[0], face.mesh[face.RIGHT_IRIS]))
            # print(f'Eye points count: {len(xvals_left_i)+len(xvals_right_i)}')
            return xvals_right_i, xvals_left_i
        else:
            # print(f'Head points count: {len(face.head_pts)}')
            xvals = list(map(lambda x: x[0], face.head_pts))
            return xvals
    
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
        1. uses single camera to collect 9 calibration images.
        2. if checkerboard of check_dims dimensions is detected, 2 and 3d points are collected.
        3. Image with the detected corners labeled is displayed.
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
        print(f'Calibration error:\n{ret}')
        print(f"\nCamera matrix: \n{matrix}")
        print(f"\n Distortion coefficients:\b{distortion}")
        print(f"\n Rotation Vectors:\n{r_vecs}")
        print(f"\n Translation Vectors:\n{t_vecs}")
        return ret, matrix, distortion, r_vecs, t_vecs

if __name__ == '__main__':
    cameras = {'camL':0}

    # face object to hold biometrics
    face = FaceDet()
    # calibrator, where distance to camera (in.) is transformed (mm)
    monocal = MonoCalibrator()
    # updated face object with focal length and iris width (mm)
    face = monocal.stream(from_saved=True)
    print(f'Focal Length: {monocal.f_monocal}')