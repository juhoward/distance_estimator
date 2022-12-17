#!/bin/bash

# stops webcam from using autofocus
# effect is not persistent, so this script must run before calibration

# camera 1
echo "CAMERA 1"
v4l2-ctl -d /dev/video0 --list-ctrls
echo ""
echo "Disabling camera 1 auto-focus"
v4l2-ctl -d /dev/video0 --set-ctrl=focus_auto=0

# camera 2
echo "CAMERA 2"
v4l2-ctl -d /dev/video2 --list-ctrls 
echo ""
echo "Disabling camera 1 auto-focus"
v4l2-ctl -d /dev/video2 --set-ctrl=focus_auto=0
