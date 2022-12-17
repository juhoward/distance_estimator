from sre_constants import SUCCESS
from flask import (Flask, render_template, Response, request)
import cv2 as cv
import os
import datetime, time
from threading import Thread
from mono_calibrate import MonoCalibrator
import numpy as np
import logging


global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('./calibration/imgs/')
except OSError as error:
    pass

app = Flask(__name__)
cam = cv.VideoCapture(0)
calibrator = MonoCalibrator()

def to_bytes(frame):
    ret, buffer = cv.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return frame

def gen_frames():
    global capture, calibrator
    cnt = 0
    p = os.path.join('calibration', 'imgs')
    while True:
        success, frame = cam.read()
        if not success:
            break
        if capture:
            capture = 0           
            img = calibrator.get_mono_calibration_data(p, frame)
            if not img is None:
                print(f'detections: {len(calibrator.points2D)}')
                if len(calibrator.points2D) > 3:
                    camera_intrinsics = calibrator.mono_calibrate()
                    # get mean of focal lengths in x and y dimensions
                    f = (camera_intrinsics[1][0][0] + camera_intrinsics[1][1][1]) / 2
                    calibrator.f_monocal = f
                    print(f'focal length: {calibrator.f_monocal}')
                img = to_bytes(img)
                # concat frame one-by-one and show result
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n')
            else: 
                frame = to_bytes(frame)
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            frame = to_bytes(frame)
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,cam
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                cam.release()
                cv.destroyAllWindows()
                
            else:
                cam = cv.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv.VideoWriter_fourcc(*'XVID')
                out = cv.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')

if __name__ == "__main__":

    app.run(host='0.0.0.0', port=5000, debug=True)