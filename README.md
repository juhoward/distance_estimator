# Distance Estimator

This distance estimator will return a single individual's distance to the camera after a 3-step calibration is complete.

## Installation

These procedures assume that miniconda is already installed.

Clone the repo, this example uses ssh
`git clone DigitalOptometrics@vs-ssh.visualstudio.com:v3/DigitalOptometrics/DigitalOptometrics/distance_estimator`

Enter the directory, then
`conda env create -f test.yaml`
and 
`conda activate test`

## Running the test script

Running this command
`python distance_estimator.py --stream --calibrate`

will initiate the calibration

## Calibration Step 1

- Present an ID card horizontally to the camera, close enough so that the edges of the card are just within the view area of the screen.
- Hold it there while pressing the "i" key.
- Once the screen freezes, stop holding the "i" key and observe the depth map. Press any key to continue.
- Observe the cropped card in black and white. Press any key to continue.

If the cropped image captured the id card, proceed. If not, press "i" again to capture another image of the ID card.

## Calibration Step 2

- Place the ID card horizontally next to one of your eyes while you are at a 1 ft. - 2ft. distance.
- Press and hold the spacebar while looking at the camera so that your irises are fully exposed.
- When enough images are captured, the command prompt will begin printing the directions for the next step

A series of images will be captured, the goal is to use the card's width to measure your iris and the width of your head.

## Calibration Step 3

To calibrate the camera, a series of pictures will be taken where a calibration target will be detected and the results of the detection displayed.
Present the target to the camera such that most of the viewing area is taken up by the target. It is a best practice to have a picture from the left, right, top, bottom, and center of the viewing area, but this not necessary for a good calibration.

- Display the checkerboard calibration target on your phone horizontally.
- Press the "c" key when the target is in the desired location.

Once the calibration procedure is complete, the live feed will begin showing relevant facial landmarks and the estimated distance to the camera.

