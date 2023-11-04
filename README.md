# ROV Computer Vision

## Calibration
1. Place uncalibrated checkerboard images in a directory `data/calibration_imgs/original`
2. While in the `X16-CV` directory, run `python calibration/calibrate.py`. This will display the calibrated images, write then to a new directory, and create a calibration file
3. To examine the results, run `python calibration/display.py`. This will show the calibrated and uncalibrated images side by side. 
