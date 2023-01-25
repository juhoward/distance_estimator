import cv2 as cv
from face import FaceDet
from detectors.body import PersonDetector
from detectors.card import CardDetector
from detectors.depth_midas import DepthEstimator
from statistics import median
import numpy as np
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
                 camera_ids:dict,
                 face,
                 checker_dims):
        self.camera_ids = camera_ids
        self.face = face
        # credit card width (mm)
        self.w_card = 85.6
        # focal length
        self.f_monocal = 0
        # variables for removing lense distortion
        self.camMat = None
        self.optimalMat = None
        self.roi = None
        # mediapipe 
        self.detector= PersonDetector(self.face)
        self.cameras = {}
        for idx, (name, id) in enumerate(camera_ids.items()):
            if idx == 0:
                self.cameras[name] = cv.VideoCapture(id)
                # turns on autofocus
                self.cameras[name].set(cv.CAP_PROP_AUTOFOCUS, 1)
                self.h = int(self.cameras[name].get(4))
                self.w = int(self.cameras[name].get(3))
        # Vector for 3D points
        self.points3D = []
        # Vector for 2D points
        self.points2D = []
        # counter
        self.cnt = 0
        # checkerboard dims
        self.checker_dims = checker_dims
        # save grayscale frame
        self.grayFrame = None


    def stream(self, card_detector, depth_estimator):
        '''
        Takes user through 3-step procedure to measure a subject's iris, estimate
        the camera's focal length, and identify and remove lense distortion.
        User must have an ID card (credit card, medical ID, state ID)
        '''
        w_irises = []
        print('Step 1 - ID Card')
        print('press and hold "i" when the ID card is near camera and in focus...')
        while True:
            for name, cam in self.cameras.items():
                ok, frame = cam.read()
                if not ok:
                    break
                cv.imshow('calibration', frame)
            # press "q" to quit
            if cv.waitKey(1) & 0xff == ord('q'):
                cam.release()
                cv.destroyAllWindows()
                print(f'Focal Length: {self.f_monocal}')
                break
            # Step 1 -- capture ID card features
            elif cv.waitKey(1) & 0xff == ord('i'):
                print('autofocus engaged.')
                # remove lense distortion from frame
                # undist = self.undistort(frame)
                self.detect_card_example(frame, depth_estimator, card_detector)
                print('\n\nStep 2 - Biometrics')
                print('Measuring iris width based on dimensions of a credit card')
                print('press and hold "spacebar" when 12 in. - 24 in. from camera ...')

            # Step 3 -- re-id card, detect iris, collect measurements
            elif cv.waitKey(1) & 0xff == ord(' '):
                # remove lense distorion from frame
                # undist = self.undistort(frame)
                # reidentify credit card and get its dimensions
                valid, h, w = self.reidentify_card(frame, card_detector)
                # assign width to card_w_pix
                self.card_w_pix = w
                if valid == True:
                    # estimate iris width
                    output = self.detect(self.face, self.detector, frame)
                    if output[0] == True:
                        self.cnt +=1
                        w_iris = output[1]
                        w_irises.append(w_iris)
                        pt1 = self.face.mesh[self.face.HEAD[0]]
                        pt2 = self.face.mesh[self.face.HEAD[1]]
                        self.face.get_headw(pt1, pt2, logging=True)
                    if self.cnt > 29:
                        # update width of the iris (mm)
                        self.face.w_iris = median(w_irises)
                        # update width of head based on iris (mm)
                        self.face.head_w = median(self.face.head_measurements)

                        # turn off autofocus in preparation for the next step
                        cam.set(cv.CAP_PROP_AUTOFOCUS, 0)
                        print('autofocus disengaged.')
                        print('Step 3, Camera calibration')
                        print('Take 5 images of the calibration target to calibrate the camera.')
                        print('Press and hold "c" when checkerboard calibration target is near camera...')

            # Step 3 -- camera calibration
            elif cv.waitKey(1) & 0xff == ord('c'):
                # collect new calibration images
                self.get_mono_calibration_data(frame)
                if len(self.points2D) > 4:
                    # reset img count
                    self.cnt = 0
                    # intrinsic matrix is at index 1 of this tuple
                    camera_intrinsics = self.mono_calibrate()
                    self.camMat = camera_intrinsics[1]
                    self.dist = camera_intrinsics[2]
                    # get mean of focal lengths in x and y dimensions
                    f = (camera_intrinsics[1][0][0] + camera_intrinsics[1][1][1]) / 2
                    self.f_monocal = f
                    # remove distortion from frame
                    h, w = frame.shape[:2]
                    # distortion coefficients are at index 2
                    self.optimalMat, self.roi = cv.getOptimalNewCameraMatrix(camera_intrinsics[1], camera_intrinsics[2], (w,h), 1, (w,h))
                    # cam.set(cv.CAP_PROP_AUTOFOCUS, 1)
                    # cue to user to move on to next procedure
                    print('Calibration complete!')
                    print(f'credit card-based iris diameter (mm): {self.face.w_iris}')
                    print(f'iris-based head width (mm): {self.face.head_w}')
                    for name in self.cameras:
                        self.cameras[name].release()
                    cv.destroyAllWindows()
                    return self.face
    def undistort(self, frame):
        '''
        uses camera intrinsics to remove lense distortion from captured frame.
        returns the undistorted frame, cropped to remove empty pixels
        '''
        # remove lense distorion from frame
        undist = cv.undistort(frame, self.camMat, self.dist, None, self.optimalMat)
        # crop image to remove empty pixels
        x, y, w, h = self.roi
        undist = undist[y:y+h, x:x+w]
        return undist

    def detect_card_example(self, frame, depth_estimator, card_det):
        '''
        uses monocular depth estimator to localize credit card, 
        then uses card detector to capture card features.
        '''
        depth_frame = depth_estimator.predict(frame)
        depth_frame = card_det.to_video_frame(depth_frame)
        cv.imshow('Depth', depth_frame)
        cv.waitKey(0)
        boundaries = card_det.detect_lines(depth_frame)
        if boundaries:
            print(f'Card captured.\n Lines detected: {boundaries}')
            card_det.crop_img(boundaries, frame)
            card_det.get_obj_features(card_det.img_obj)
            cv.imshow('Cropped', card_det.img_obj)
            cv.waitKey(3000)
            cv.destroyWindow('Cropped')
            cv.destroyWindow('Depth')
            print(f'{len(card_det.keypoints_obj)} features stored.')

    def reidentify_card(self, frame, card_det):
        '''
        uses card detector to re-identify a credit card.
        '''
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        scene_corners = card_det.reidentify(frame)
        if scene_corners is not None:
            valid, h, w = card_det.validate(scene_corners)
            if valid == True:
                self.cnt += 1
                card_det.show_match(scene_corners, frame, (h,w), self.cnt, save_img=False)
            return valid, h, w
        else: 
            return False, 0, 0

    def detect(self, face, detector, frame):
        '''
        uses mediapipe detectors to localize iris, updates face object, and returns
        focal length estimates based on iris width.
        '''
        face.mesh = None        
        detector.findIris(frame)
        output = []
        if not face.mesh is None:
            output.append(True)
            # transforms to real width and updates face mesh
            face.update_iris_width(self.w_card, self.card_w_pix)                             
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
        return img

    def xvals(self, face):
        '''
        collect x values from iris keypoint detections. x values are used for disparity
        '''
        xvals_left_i = list(map(lambda x: x[0], face.mesh[face.LEFT_IRIS]))
        xvals_right_i = list(map(lambda x: x[0], face.mesh[face.RIGHT_IRIS]))
        # print(f'Eye points count: {len(xvals_left_i)+len(xvals_right_i)}')
        return xvals_right_i, xvals_left_i

    
    def mono_calibration_data_reader(self, img_dir):
        '''
        reads previously collected calibration images.
        '''
        imgs = glob.glob('./calibration/mono_imgs/*.png')
        return (cv.cvtColor(cv.imread(im), cv.COLOR_BGR2GRAY) for im in imgs)

    def find_corners(self, img_reader):
        '''
        corner detector for the checkerboard calibration target.
        '''
        criteria = (cv.TERM_CRITERIA_EPS +
                    cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 3D points real world coordinates
        objectp3d = np.zeros((1, 
                            self.checker_dims[0] * self.checker_dims[1],
                            3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:self.checker_dims[0],
                                    0:self.checker_dims[1]].T.reshape(-1, 2)
        for img in img_reader:
            ret, corners = cv.findChessboardCorners(
                            img, self.checker_dims,
                            cv.CALIB_CB_ADAPTIVE_THRESH
                            + cv.CALIB_CB_FAST_CHECK +
                            cv.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                self.grayFrame = img
                self.points3D.append(objectp3d)
                # (11,11) is the corner search window size
                # (-1,-1) is a null value, but can be adjusted if singularities in
                # the autocorrelation matrix are frequent. This is not likely.
                corners2 = cv.cornerSubPix(
                            self.grayFrame, corners, (11, 11), (-1, -1), criteria)
                self.points2D.append(corners2)
                print(f'{10 - len(self.points2D)} more calibration photos to take!')
    
    def get_mono_calibration_data(self, frame, save=True):
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
                            self.checker_dims[0] * self.checker_dims[1],
                            3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:self.checker_dims[0],
                                    0:self.checker_dims[1]].T.reshape(-1, 2)
        self.grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv.findChessboardCorners(
                        self.grayFrame, self.checker_dims,
                        cv.CALIB_CB_ADAPTIVE_THRESH
                        + cv.CALIB_CB_FAST_CHECK +
                        cv.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            if save == True:
                self.cnt += 1
                fname = './calibration/mono_imgs/' + str(self.cnt) + '.png'
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
                                            self.checker_dims,
                                            corners2, ret)
            cv.imshow('Target Detected', image)

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
        return ret, matrix, distortion, r_vecs, t_vecs

if __name__ == '__main__':
    ############################# do not remove
    # necessary to initialize gui by using imshow before card detector is initialized
    impath = './calibration/mono_imgs/1.png'
    img = cv.imread(impath)
    cv.imshow('img', img)
    cv.waitKey(1)
    cv.destroyWindow('img')
    ############################# end do not remove                                                                                                                                                                                                                                                                                                                                                          
    # monocalibrator potentially accepts multiple cameras
    # this is only to be compatible wth stereo_depth_calibrated script
    cameras = {'camL':0}

    # load midas depth estimator
    model_type = "DPT_Large" 
    estimator = DepthEstimator(model_type)
    # load card detector
    card_detector = CardDetector('SIFT')
    # face object to hold biometrics
    face = FaceDet()
    # calibrator object that calculates focal length and iris diameters
    monocal = MonoCalibrator(cameras, face, (6,9))
    # updated face object with focal length and iris width (mm)
    face = monocal.stream(card_detector, estimator, from_saved=True)
    print(f'Focal Length: {monocal.f_card}')
    print(f'Iris width: {face.w_iris}')
    print(f'Head width: {face.head_w}')
