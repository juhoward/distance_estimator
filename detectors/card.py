from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from math import (sin, cos, dist)
from .depth_midas import DepthEstimator
from glob import glob

def showInMovedWindow(winname, img, x, y):
    '''
    creates a named window and moves it to a location.
    the window that is shown cannot be moved.
    '''
    # Create a named window
    cv.namedWindow(winname, cv.WINDOW_AUTOSIZE)        
    # Move it to (x,y)
    cv.moveWindow(winname, x, y)  
    cv.imshow(winname,img)

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)
    return qx, qy

def get_threshold_vals(img, sigma = .33):
    '''
    returns the upper and lower threshold values for an image based on the median pixel value.
    '''
    med = np.median(img)
    lower = int(max(0, (1 - sigma) * med))
    upper = int(min(255, (1 - sigma) * med))
    return lower, upper


class CardDetector(object):
    def __init__(self, method):
        if method == 'SIFT':
            self.detector = cv.SIFT_create()
            FLANN_INDEX_KDTREE = 1
            self.index_params = {'algorithm':FLANN_INDEX_KDTREE, 'trees':5}
        else:
            self.detector = cv.ORB_create()
            FLANN_INDEX_LSH = 6
            self.index_params = {'algorithm':FLANN_INDEX_LSH, 
                            'table_number': 6, #12,
                            'key_size': 12, #20,
                            'multi_probe_level': 1} #2
        self.keypoints_obj = None
        self.descriptors_obj = None
        self.keypoints_scene = None
        self.good_matches = []
        self.img_obj = None
        # credit card's aspect ratio (height / width)
        self.card_h = 53.98
        self.card_w = 85.6
        # credit card in scene
        self.card_location = None
        self.tp = 0
        self.fp = 0

    def stream(self, estimator, reidentify=False):
        camera = cv.VideoCapture(0)
        # turns on autofocus
        camera.set(cv.CAP_PROP_AUTOFOCUS, 1)
        print('Press "c" to when example object is within the green box')
        print('Press "q" to exit.')
        cnt = 0
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            h = frame.shape[0]
            w = frame.shape[1]
            # copy frame so you don't modify the original capture
            highlight = cv.rectangle(frame.copy(), (int(w*.05), int(h*.1)), (int(w * .95), int(h*.9)), (0,255,0), 2)
            cv.imshow('Credit Card Capture', highlight)
            # make the window pop up in specific location on screen
            # showInMovedWindow('Credit Card Capture', highlight, 1400, 200)
            if cv.waitKey(1) & 0xff == ord('q'):
                break
            elif cv.waitKey(1) & 0xff == ord('c'):
                depth_frame = estimator.predict(frame)
                depth_frame = self.to_video_frame(depth_frame)
                cv.imshow('Depth', depth_frame)
                boundaries = self.detect_lines(depth_frame)
                if boundaries:
                    print(f'Card captured.\n Lines detected: {boundaries}')
                    self.crop_img(boundaries, frame)
                    self.get_obj_features(self.img_obj)
                    cv.imshow('Cropped', self.img_obj)
                    cv.waitKey(3000)
                    cv.destroyWindow('Cropped')
                    cv.destroyWindow('Depth')
                    print(f'{len(self.keypoints_obj)} features stored.')
                    print('Press "r" to re-identify card in scene.')
            elif cv.waitKey(1) & 0xff == ord('r'):
                if reidentify == False:
                    reidentify = True
                    print('Re-ID engaged.')
                else:
                    reidentify = False
            if reidentify == True:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                scene_corners = self.reidentify(frame)
                if scene_corners is not None:
                    valid, h, w = self.validate(scene_corners)
                    if valid == True:
                        cnt += 1
                        self.show_match(scene_corners, frame, (h,w), cnt)
        print(f'Precision: {self.get_precision()}')
        camera.release()
        cv.destroyAllWindows()

    def get_precision(self):
        return round((self.tp / (self.tp + self.fp)) * 100, 2)

    def validate(self, scene_corners, err_margin = .2):
        '''
        validates the re-id card's aspect ratio matches a real card's aspct ratio.
        scene corners: output of perspective transform, subpixel accuracy
        err_margin: the acceptable margin of error

        returns a boolean and tuple of (h, w)
        '''
        tl = (scene_corners[0,0,0] + self.img_obj.shape[1], scene_corners[0,0,1])
        tr = (scene_corners[1,0,0] + self.img_obj.shape[1], scene_corners[1,0,1])
        br = (scene_corners[2,0,0] + self.img_obj.shape[1], scene_corners[2,0,1])
        bl = (scene_corners[3,0,0] + self.img_obj.shape[1], scene_corners[3,0,1])
        # left side length
        h1 = dist(tl, bl)
        # right side legnth
        h2 = dist(tr, br)
        # top length
        w1 = dist(tl, tr)
        # bottom length
        w2 = dist(bl, br)
        # take averages
        w = (w1 + w2) / 2
        h = (h1 + h2) / 2

        if w == 0:
            return False, 0
        # I flip the scene h and w values so that validate works
        aspect_ratio = self.card_h / self.card_w
        # calculate the percentage of the aspect ratio we deem acceptable
        val = aspect_ratio * err_margin
        scene_ratio = h / w
        # if difference is less than margin of erro, accept the re-id coordinates
        if abs(aspect_ratio - scene_ratio) < val:
            self.tp += 1
            return True, h, w
        else:
            self.fp += 1
            print(f'False Positive\ndifference: {abs(aspect_ratio - scene_ratio)}\nval:{val}')
            return False, 0, 0

    def get_obj_features(self, img_obj):
        self.keypoints_obj, self.descriptors_obj = self.detector.detectAndCompute(img_obj, None)

    def reidentify(self, img_scene, ratio_thresh=.5, min_matches=8):
        keypoints_scene, descriptors_scene = self.detector.detectAndCompute(img_scene, None)
        self.keypoints_scene = keypoints_scene
        #-- Step 2: Matching descriptor vectors with a FLANN based matcher
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(self.descriptors_obj, descriptors_scene, 2)
        self.good_matches.clear()
        #-- Filter matches using the Lowe's ratio test
        for m,n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                self.good_matches.append(m)
        if len(self.good_matches) > min_matches:
            #-- Localize the object
            obj = np.empty((len(self.good_matches),2), dtype=np.float32)
            scene = np.empty((len(self.good_matches),2), dtype=np.float32)
            for i in range(len(self.good_matches)):
                #-- Get the keypoints from the good matches
                obj[i,0] = self.keypoints_obj[self.good_matches[i].queryIdx].pt[0]
                obj[i,1] = self.keypoints_obj[self.good_matches[i].queryIdx].pt[1]
                scene[i,0] = keypoints_scene[self.good_matches[i].trainIdx].pt[0]
                scene[i,1] = keypoints_scene[self.good_matches[i].trainIdx].pt[1]
            H, _ =  cv.findHomography(obj, scene, cv.RANSAC, 10)
            if H is not None:
                #-- Get the corners from the image_1 ( the object to be "detected" )
                obj_corners = np.empty((4,1,2), dtype=np.float32)
                obj_corners[0,0,0] = 0
                obj_corners[0,0,1] = 0
                obj_corners[1,0,0] = self.img_obj.shape[1]
                obj_corners[1,0,1] = 0
                obj_corners[2,0,0] = self.img_obj.shape[1]
                obj_corners[2,0,1] = self.img_obj.shape[0]
                obj_corners[3,0,0] = 0
                obj_corners[3,0,1] = self.img_obj.shape[0]
                try:
                    # localize object in scene
                    scene_corners = cv.perspectiveTransform(obj_corners, H)
                except:
                    print(f"possible homography failure. H type: {type(H)} \tsize: {H.size}\tH: {H}\tgood match count: {len(self.good_matches)}")
                return scene_corners

    def detect_lines(self, img, threshold=10):
        '''
        uses the opencv native line segment detector to find candidate lines with which to crop the img_obj.
        img : a 3-channel image that is assumed to be the midas depth output
        TODO: test foreground background subtraction mask instead of MiDaS outputs.
        '''
        cnt = 0
        # Create default parametrization LSD
        lsd = cv.createLineSegmentDetector(0)

        # Detect lines in the image
        # Position 0 of the returned tuple are the detected lines
        lines = lsd.detect(img[:,:,0])[0] 
        # remove a dimension
        lines = lines[:,0]
        # restrict domain of potential line points to those within the domain of the image.
        # negative values or values beyond the limits of the img are sometimes returned.
        lines = np.clip(lines, 0, img.shape[1])

        # Filter out the lines whose length is lower than the threshold
        # euclidean distances
        # dx = lines[:, 2] - lines[:, 0]
        # dy = lines[:, 3] - lines[:, 1]
        # lengths = np.sqrt(dx * dx + dy * dy)
        # mask = lengths >= threshold
        # lines = lines[mask]

        # filter for horizontal and vertical lines
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        dx += .001
        slopes = np.abs(dy / dx)
        horizontal = slopes < .7
        vertical = slopes > (max(slopes) *.10)
        hlines = lines[horizontal]
        vlines = lines[vertical]

        print('vertical lines:', len(vlines))
        print('horizontal lines:', len(hlines))
        # if all 4 edges are detected
        if len(vlines) >= 2 and len(hlines) >= 2:
            # find the most distant horizontal points
            hpts = self.farthest(hlines, horizontal=True)
            # find the most distant vertical points
            vpts = self.farthest(vlines, horizontal=False)
            # returns y1, y2, x1, x2 for cropping
            return hpts[0][1], hpts[1][1], vpts[0][0], vpts[1][0]
        # if vertical lines are detected
        elif len(vlines) >= 2:
            vpts = self.farthest(vlines, horizontal=False)
            # returns x1, x2 for cropping
            return None, None, vpts[0][0], vpts[1][0]
        # if horizontal lines are detected
        elif len(hlines) >= 2:
            hpts = self.farthest(hlines, horizontal=True)
            # returns y1, y2 for cropping
            return hpts[0][1], hpts[1][1], None, None
        else:
            cnt += 1
            drawn_img = lsd.drawSegments(img.copy(), lines)
            cv.imwrite(f'./results/Lines_{cnt}.png',drawn_img)
            drawn_img = lsd.drawSegments(img.copy(), vlines)
            cv.imwrite(f'./results/vertical_{cnt}.png',drawn_img)
            drawn_img = lsd.drawSegments(img.copy(), hlines)
            cv.imwrite(f'./results/horizontal_{cnt}.png',drawn_img)

    def farthest(self, lines, horizontal=True):
        '''
        given a list of candidate lines, uses the absolute difference in the x or y dim
        and the points representing the farthest lines.
        Points are intended to set cropping boundaries of the img_obj image
        '''
        max_dist = 0
        max_pts = []
        for i in lines:
            x1 = i[0]
            y1 = i[1]
            for j in lines:
                x2 = j[0]
                y2 = j[1]
                if horizontal:
                    d = abs(y2 - y1)
                else:
                    d = abs(x2 - x1)
                if d > max_dist:
                    max_dist = d
                    pt1 = tuple(map(lambda x: int(x), (x1,y1)))
                    pt2 = tuple(map(lambda x: int(x), (x2,y2)))
                    max_pts = [pt1, pt2]
        # sorting least to greatest to make cropping easier
        if horizontal:
            # sort by y values if horizontal
            max_pts.sort(key=lambda x: x[1])
            return max_pts
        else:
            # sort by x values if vertical
            max_pts.sort(key=lambda x: x[0])
            return max_pts

    def crop_img(self, boundaries, img_obj):
        '''
        uses the boundaries provided by line detector to crop the card image.
        '''
        y1, y2, x1, x2  = boundaries
        # horizontal and vertical boundaries are located
        if boundaries[0] and boundaries[2] is not None:
            cropped = img_obj[y1:y2, x1:x2]
        # only horizontal boundaries
        elif boundaries[0] is not None:
            cropped = img_obj[y1:y2, :]
        # only vertical boundaries
        elif boundaries[2] is not None:
            cropped = img_obj[:, x1:x2]
        cropped = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        self.img_obj = cropped

    def to_video_frame(self, img):
        ''' 
        transforms midas depth frame to a video frame.
        '''
        output = img.astype(np.uint8)
        # change contrast
        output *= 5
        # brightness
        output += 10
        return cv.merge([output,output,output])

    def show_match(self, scene_corners, img_scene, distance, cnt, save_img=True):
        '''
        A visualization tool to show the localization of a credit card and matched pixels.
        uses the corners from a projective transformation to localize the card in a given scene image.
        Drawing matched points is completed by the drawMatches function, while the scene corners are used
        to draw the localization lines.
        A good localization is dependent on a high quality img_obj as a reference.
        '''
        #-- Draw keypoint matches
        img_matches = np.empty((max(self.img_obj.shape[0], img_scene.shape[0]), self.img_obj.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
        cv.drawMatches(self.img_obj, self.keypoints_obj, img_scene, self.keypoints_scene, self.good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #-- Draw lines between the corners (the mapped object in the scene - image_2 )
        # coordinates must be integers to display them
        display = scene_corners.astype(int)
        tl = (display[0,0,0] + self.img_obj.shape[1], display[0,0,1])
        tr = (display[1,0,0] + self.img_obj.shape[1], display[1,0,1])
        br = (display[2,0,0] + self.img_obj.shape[1], display[2,0,1])
        bl = (display[3,0,0] + self.img_obj.shape[1], display[3,0,1])
        cv.line(img_matches, tl, tr, (255,0,0), 4)
        cv.line(img_matches, bl, br, (255,0,0), 4)
        cv.line(img_matches, tl, bl, (0,255,0), 4)
        cv.line(img_matches, tr, br, (0,255,0), 4)
        cxt = (tl[0] + tr[0]) // 2
        # cxb = (bl[0] + br[0]) // 2
        # cx = int((cxt + cxb) / 2)
        # cyt = (tl[1] + tr[1]) // 2
        # cyb = (bl[1] + br[1]) // 2
        # cy = int((cyt + cyb) / 2)
        cv.circle(img_matches, (tl[0], tl[1]),2,(255,0, 255), 2, cv.LINE_AA)
        cv.circle(img_matches, (tr[0], tr[1]),2,(255,0, 255), 2, cv.LINE_AA)
        cv.circle(img_matches, (bl[0], bl[1]),2,(255,0, 255), 2, cv.LINE_AA)
        cv.circle(img_matches, (br[0], br[1]),2,(255,0, 255), 2, cv.LINE_AA)
        h_message = str(round(distance[0], 2))
        w_message = str(round(distance[1], 2))
        cv.putText(img_matches, h_message, (tl[0],(tl[1] + bl[1]) // 2),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv.LINE_AA)
        cv.putText(img_matches, w_message, (cxt,tl[1]),

                                cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv.LINE_AA)
        # for pt, m in zip([tl, tr, bl,br], ['tl', 'tr', 'bl', 'br']):
        #         message = m + ' ('+str(pt[0])+', '+str(pt[1])+')'
        #         cv.putText(img_matches, message, (pt[0],pt[1]),
        #                     cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv.LINE_AA)
        if save_img:
            cv.imwrite(f'./Card_Detection_{cnt}.png', img_matches)
            print(f'Capture saved: ./Card_Detection_{cnt}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN')
    parser.add_argument('--cards', help='Path to card pictures', default='./cards2/')
    parser.add_argument('--scenes', help='Path to card re-id scenes.', default='./scenes_forehead/')
    parser.add_argument('--method', help='keypoint detectors: SIFT, ORB', default='SIFT')
    parser.add_argument('--stream', help='card detection in video stream', action=argparse.BooleanOptionalAction)
    parser.add_argument('--camera_ids', help='A dictionary with the name and busid of device', default={'camL':0})
    args = parser.parse_args()
    ################################ do not remove
    # necessary to initialize gui before card detector is initialized
    objects = sorted(glob(args.cards + '*.jpg'))
    for o in objects[:1]:
        img = cv.imread(o)
        cv.imshow('img', img)
        cv.waitKey(1)
        cv.destroyWindow('img')
    ################################ end do not remove
    if args.stream == True:
        # load midas depth estimator
        model_type = "DPT_Large" 
        estimator = DepthEstimator(model_type)
        # load card detector
        card_finder = CardDetector('SIFT')
        card_finder.stream(estimator)
    
    ######################## bug related to number of channels
    else:
        print('reading saved test set images...')
        objects = sorted(glob(args.cards + '*.jpg'))
        scenes = sorted(glob(args.scenes + '*.jpg'))
        # load midas depth estimator
        model_type = "DPT_Large" 
        estimator = DepthEstimator(model_type)
        # load card detector
        card_finder = CardDetector('SIFT')

        cnt = 0
        for obj, scn in zip(objects, scenes):
            print(obj)
            print(scn)
            img_obj = cv.imread(obj, cv.IMREAD_GRAYSCALE)
            img_obj_color = cv.imread(obj)
            img_scene = cv.imread(scn, cv.IMREAD_GRAYSCALE)
            if img_obj is None or img_scene is None:
                print('Could not open or find the images!')
                continue

            # get depth estimate
            depth_frame = estimator.predict(img_obj_color)
            depth_frame = card_finder.to_video_frame(depth_frame)
            # detect card boundaries
            boundaries = card_finder.detect_lines(depth_frame)
            if boundaries:
                card_finder.crop_img(boundaries, img_obj)
                y1, y2, x1, x2  = boundaries
                # horizontal and vertical boundaries are located
                if boundaries[0] and boundaries[2] is not None:
                    cropped = img_obj[y1:y2, x1:x2]
                    card_finder.get_obj_features(cropped)
                    scene_corners = card_finder.reidentify(img_scene)
                    cnt += 1
                    card_finder.show_match(scene_corners, img_scene, cnt)
                # only horizontal boundaries
                elif boundaries[0] is not None:
                    cropped = img_obj[y1:y2, :]
                    card_finder.get_obj_features(cropped)
                    scene_corners = card_finder.reidentify(img_scene)
                    cnt += 1
                    card_finder.show_match(scene_corners, img_scene, cnt)
                # only vertical boundaries
                elif boundaries[2] is not None:
                    cropped = img_obj[:, x1:x2]
                    card_finder.get_obj_features(cropped)
                    scene_corners = card_finder.reidentify(img_scene)
                    cnt += 1
                    card_finder.show_match(scene_corners, img_scene, cnt)
            else:
                print(f'No detection: {obj}, {scn}')