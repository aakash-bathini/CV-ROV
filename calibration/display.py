import cv2
import os
import glob

# Load the two images
image1_files = glob.glob('data\\dual\\cal\\*.jpeg')  # Change to your image directory

for distorted_file in image1_files:
    filename = os.path.basename(distorted_file).split('\\')[-1]
    undistorted_file = "data\\dual_cal\\undistorted_"+filename
    print(distorted_file, undistorted_file)

    image1 = cv2.imread(distorted_file)
    image2 = cv2.imread(undistorted_file)

    # Get the heights of both images
    height1, height2 = image1.shape[0], image2.shape[0]

    # Find the maximum height
    max_height = max(height1, height2)

    # Resize the images to have the same height
    image1 = cv2.resize(image1, (int(image1.shape[1] * (max_height / height1)), max_height))
    image2 = cv2.resize(image2, (int(image2.shape[1] * (max_height / height2)), max_height))

    # Create a new image by concatenating the two images horizontally
    combined_image = cv2.hconcat([image1, image2])

    # Display the combined image
    cv2.imshow("Side-by-Side Images", combined_image)

    # Wait for a key press and then close the window
    cv2.waitKey(0)

cv2.destroyAllWindows()
