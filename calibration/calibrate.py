import numpy as np
import cv2
import glob
import os

# Define the chessboard pattern parameters
chessboard_size = (9, 6)  # Change to your chessboard size
square_size = 1.0  # Change to your square size in your units (e.g., cm)

# Prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size

# Lists to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Create a list of calibration images
image_files = glob.glob('data\\calibration_imgs\\original\\*.jpeg')  # Change to your image directory

# Loop through each image and find chessboard corners
for image_file in image_files:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If corners are found, add object points and image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the calibration parameters to a file (optional)
calibration_data = {
    'camera_matrix': mtx,
    'distortion_coefficients': dist,
}
np.save('calibration\\calibration_data.npy', calibration_data)


os.makedirs("data\\calibration_imgs\\calibrated", exist_ok=True)
# Undistort images
for image_file in image_files:
    img = cv2.imread(image_file)
    undistorted_img = cv2.undistort(img, mtx, dist)
    # Save undistorted image
    undistorted_filename = 'data\\calibration_imgs\\calibrated\\calibrated_' + os.path.basename(image_file).split('\\')[-1]
    cv2.imwrite(undistorted_filename, undistorted_img)

    # Show an example of the undistorted image (optional)
    cv2.imshow('Undistorted Image', undistorted_img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
