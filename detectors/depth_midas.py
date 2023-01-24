import cv2
import torch
import numpy as np


class DepthEstimator(object):
    def __init__(self, model_type):
        self.model_type = model_type
        print(f'Loading model: {self.model_type}')
        self.estimator = torch.hub.load("intel-isl/MiDaS", self.model_type)
        print('Loading transforms...')
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.estimator.to(self.device)
        self.estimator.eval()

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.estimator(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        output = prediction.cpu().numpy()
        return output


class VidStream(object):
    ''' a wrapper for OpenCV that accepts a depth estimator object'''
    def __init__(self, estimator, src=None, output=None):
        self.estimator = estimator
        self.video = cv2.VideoCapture(src)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        # FPS = 1/X, X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)
        self.w = int(self.video.get(3))
        self.h = int(self.video.get(4))
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.writer = cv2.VideoWriter(str(output), self.fourcc, 20, (self.w, self.h))
        self.status = None
        self.frame = None
        self.cnt = 0

    def update(self):
        while True:
            if self.video.isOpened() == False:
                print('Error opening file.')
            
            if self.video.isOpened():
                self.status, self.frame = self.video.read()
                self.cnt += 1
                print(f'Frame: {self.cnt}')
                if self.status == True:
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        self.video.release()
                        self.writer.release()
                        break
                    self.write_output()
                else:
                    self.video.release()
                    self.writer.release()
                    break
            else:
                print(f'Sucessfully read {self.cnt} out of {self.video.get(7)} frames.')
                self.video.release()
                self.writer.release()
                break
        cv2.destroyAllWindows()
    
    def write_output(self):
        prediction = self.estimator.predict(self.frame)
        output = prediction.astype(np.uint8)
        three_c = cv2.merge([output,output,output])
        self.writer.write(three_c)



if __name__ == '__main__':
    # load depth estimator
    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    vid = "/home/digitalopt/proj/face_depth/webcam_video.mp4"
    vid2 = "/home/digitalopt/proj/face_depth/card_20_10_5.mp4"
    vid3 = "/home/digitalopt/proj/face_depth/10ft.mp4"
    vid4 = "/home/digitalopt/proj/face_depth/occlusion_1_10.mp4"
    output = '/home/digitalopt/proj/face_depth/midas_output.avi'
    midas = DepthEstimator(model_type)
    video_stream = VidStream(midas, vid4, output)
    video_stream.update()
