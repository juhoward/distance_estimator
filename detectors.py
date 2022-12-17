import numpy as np
import mediapipe as mp
import cv2


class PersonDetector(object):
    """
    Find faces in realtime using the light weight model provided in the mediapipe
    library.
    """

    def __init__(self, face, minDetectionCon=0.2):
        """
        :param minDetectionCon: Minimum Detection Confidence Threshold
        """
        # face mesh indices
        self.LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

        # iris model horizontal points (left, right), vertical points (top, bottom)
        self.HEAD = [234, 454, 10, 152]
        # body pose head points
        self.BODY_HEAD = [i for i in range(11)]
        # raw coordinates for card from test data
        self.CARD = [505, 504, 675, 501]
        # mediapipe model config
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.mpface_mesh = mp.solutions.face_mesh
        # mediapipe model output
        self.results = None
        # image dimensions
        self.w = None
        self.h = None
        # face, iris points & measurements
        self.face = face
    
    def findIris(self, img):
        '''
        Detect Irises of a single person in an image. 
        Returns a point mesh. 
        '''
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self.mpface_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
            ) as face_mesh:
                self.h, self.w = img.shape[:2]
                self.results = face_mesh.process(imgRGB)
                if self.results.multi_face_landmarks:
                    mesh_points = np.array(
                        [
                            np.multiply([p.x, p.y], [self.w, self.h]) for p in self.results.multi_face_landmarks[0].landmark
                        ]
                    )
                    self.face.mesh = mesh_points

    def findBody(self, img, draw=False):
        '''
        Detect body.
        Returns list of 11 head points and the image used to detect them.
        '''
        mp_pose = mp.solutions.pose
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_drawing = mp.solutions.drawing_utils
        head_pts = []
        with mp_pose.Pose(min_tracking_confidence=0.5,
                          min_detection_confidence=0.7,
                          model_complexity=2) as pose:

            self.results = pose.process(img)
            # img.flags.writeable = True
            if draw:
                mp_drawing.draw_landmarks(
                    img,
                    self.results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            if self.results.pose_landmarks:
                head_pts.append(True)
                # head_pts[1] has the same indices as mediapipe's landmark mapping
                # head_pts.append([(0,0) for i in range(11)])
                for idx, pt in enumerate(self.results.pose_landmarks.landmark):                  
                    # include all headpoints except those defining the 
                    # eye corners (1,3), (4,6) because they reduce accuracy at 10ft
                    if idx in [0,2,5,7,8,9,10]:
                        center = np.multiply([pt.x, pt.y], [self.w, self.h])
                        head_pts.append(center)
                        # visualization
                        # center = center.astype(int)
                        # cv2.circle(img, center, 2, (255,0,255), 2, cv2.LINE_AA)
                        # message = f"{idx}"
                        # cv2.putText(img, message, (center[0], center[1]-20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1, cv2.LINE_AA)
                        
                # maybe use real 3D positions later. All points relative to center of hip joint
                # body_pts = [
                #     {
                #         'x': int(p.x * self.w),
                #         'y': int(p.y * self.h),
                #         'z': p.z,
                #         'visibility':p.visibility
                #     } for p in self.results.pose_world_landmarks.landmark]
                return img, head_pts
            else:
                print('Detector: Body not detected')
                head_pts.append(False)
                return img, head_pts

    def visualize(self, img):
        '''
        function that visualizes face mesh returned by iris detections.
        '''
        # iris visualization
        left_i = self.face.mesh[self.LEFT_IRIS].astype(int)
        right_i = self.face.mesh[self.RIGHT_IRIS].astype(int)
        self.face.l_iris['center'], self.face.l_iris['radius'] = cv2.minEnclosingCircle(left_i)
        self.face.r_iris['center'], self.face.r_iris['radius'] = cv2.minEnclosingCircle(right_i)
        center_left = np.array(self.face.l_iris['center'], dtype=np.int32)
        center_right = np.array(self.face.r_iris['center'], dtype=np.int32)
        cv2.circle(img, center_left, int(self.face.l_iris['radius']), (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(img, center_right, int(self.face.r_iris['radius']), (255,0,255), 2, cv2.LINE_AA)
        # eye outline visualization
        cv2.polylines(img, [self.face.mesh[self.LEFT_EYE].astype(int)], True, (0,255,0), 1, cv2.LINE_AA)
        cv2.polylines(img, [self.face.mesh[self.RIGHT_EYE].astype(int)], True, (0,255,0), 1, cv2.LINE_AA)

        cv2.line(img, self.face.mesh[self.HEAD[0]].astype(int), self.face.mesh[self.HEAD[1]].astype(int), (0,255,0), 1, cv2.LINE_AA)
        cv2.line(img, self.face.mesh[self.HEAD[2]].astype(int), self.face.mesh[self.HEAD[3]].astype(int), (0,255,0), 1, cv2.LINE_AA)
        # credit card points, take these out later
        # cv2.circle(img, (505,504), 1, (255,0,255), 2, cv2.LINE_AA)
        # cv2.circle(img, (675,501), 1, (255,0,255), 2, cv2.LINE_AA)
        # credit card pts 2
        # cv2.circle(img, (315,240), 1, (255,0,255), 2, cv2.LINE_AA)
        # cv2.circle(img, (402,240), 1, (255,0,255), 2, cv2.LINE_AA)
        # iris output
        self.frame = img
