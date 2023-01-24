import cv2 as cv
import numpy as np
import argparse
from face import FaceDet
from detectors.depth_midas import DepthEstimator
from detectors.card import CardDetector
from detectors.body import PersonDetector
from mono_calibrate import MonoCalibrator

class DistanceEstimator(object):
    def __init__(self, camera_parameters):
        self.camera = cv.VideoCapture(0)
        self.f_length = camera_parameters.f_monocal
        self.camMat = camera_parameters.camMat
        self.dist = camera_parameters.dist
        self.optimalMat = camera_parameters.optimalMat
        self.roi = camera_parameters.roi

    def stream(self, face, detector):
        print(f'Press "q" to exit.')
        self.cnt = 0
        while True:
            ok, frame = self.camera.read()
            if not ok:
                self.camera.release()
                break
            if cv.waitKey(1) & 0xff == ord('q'):
                self.camera.release()
                cv.destroyAllWindows()
                break
            # remove lense distorion from frame
            undist = cv.undistort(frame, self.camMat, self.dist, None, self.optimalMat)
            # crop image to remove empty pixels
            x, y, w, h = self.roi
            undist = undist[y:y+h, x:x+w]
            undist = np.ascontiguousarray(undist)
            image, distance = self.detect(undist, self.f_length, face, detector)
            cv.imshow('Distance Estimation', image)
            cv.waitKey(1)

    def detect(self, frame, f_length, face, detector):
        # clear face mesh
        face.mesh = None
        # populate face object with keypoints
        detector.findIris(frame)

        # if a face is detected, base S2c distance on iris diameter
        if not face.mesh is None:
            if len(face.mesh) > 468:
                # calculate median iris diameter (pixels)
                i_diameter = face.get_iris_diameter()
                # subject-to-camera distance (cm)
                s2c_dist = self.s2c_dist(f_length, face.w_iris, i_diameter)
                # convert from (cm) to (in)
                s2c_dist /= 2.54
                # write output to rgb frame
                message = f"S2C Distances(in): {s2c_dist}"
                messages = [message]
                self.write_messages(messages, frame)
                return frame, s2c_dist

        # if no face is detected by iris model, use holistic
        # S2C distance is based on median head width relative to iris diameter
        elif face.mesh is None:
            detector.holistic(frame)
            if not face.mesh is None:
                detector.visualize(frame, 'face')
                pt1 = face.mesh[face.HEAD[0]]
                pt2 = face.mesh[face.HEAD[1]]
                face.get_headw(pt1, pt2, logging=False)
                # subject-to-camera distance (cm)
                s2c_dist = self.s2c_dist(f_length, face.head_w, face.head_pixw)
                # convert from (cm) to (in)
                s2c_dist /= 2.54
                message = 'Iris not detected. Using holistic face mesh.'
                message2 = f"S2C Distances(in): {s2c_dist}"
                messages = [message, message2]
                self.write_messages(messages, frame)
                return frame, s2c_dist
        # if no head points detected, neither model found a person
        else:
            # print(f'No detection. Face mesh: {type(self.face.mesh)}\nHead pts: {head_pts}')
            message = 'Body not detected.'
            # depth_frame = self.to_video_frame(depth_frame)
            self.write_messages([message], frame)
        return frame, None

    def s2c_dist(self, f, w_object, w_pix):
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
        return s2c_d

    def write_messages(self, messages, frame):
        for idx, m in enumerate(messages):
            cv.putText(frame, m, (50, 50 + idx*50), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN')
    parser.add_argument('--depth_model', help='Depth model used for card localization', default="DPT_Large")
    parser.add_argument('--feature_selector', help='Card feature description method: SIFT, ORB', default='SIFT')
    parser.add_argument('--checker_dims', help='dimensions of interior checkerboard', default=(6,9))
    parser.add_argument('--calibrate', help='Initiates camera calibration', action=argparse.BooleanOptionalAction)
    parser.add_argument('--stream', help='card detection in video stream', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cameras', help='A dictionary with the name and busid of device', default={'camL':0})
    args = parser.parse_args()
    ############################# do not remove
    # necessary to initialize gui by using imshow before card detector is initialized
    img = np.zeros((50,50,1), dtype=np.uint8)
    cv.imshow('img', img)
    cv.waitKey(1)
    cv.destroyWindow('img')
    ############################# end do not remove

    # calibration procedure
    if args.calibrate:
        # select a neural depth estimator
        # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        estimator = DepthEstimator(args.depth_model)
        # load card detector
        card_detector = CardDetector(args.feature_selector)
        # face object holds face data
        face = FaceDet()
        monocal = MonoCalibrator(args.cameras, face, args.checker_dims)
        # Monocalibrator calulates camera intrinsics and updates face's iris width (mm)
        face = monocal.stream(card_detector, estimator)

        # logic to prevent abnormal iris diameters
        # 12.5 mm is a threshold for diagnosing megalocornea
        # 11.0 mm is a threshold for diagnosing microcornea
        # assume measurement error and default to mean iris width
        if face.w_iris > 12.4 or face.w_iris < 11:
            print(f'measurement outside acceptable limits: {face.w_iris}\n defaulting to 11.7 mm')
            face.w_iris = 11.7
        print('Monocular Focal Length Estimate:')
        print(f'focal length: {monocal.f_monocal}')
    # distance estimation tools
    detector = PersonDetector(face)
    d_estimator = DistanceEstimator(monocal)
    if args.stream:
        d_estimator.stream(face, detector)