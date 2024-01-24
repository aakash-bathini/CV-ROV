import cv2

# Import the image
image = cv2.imread('image2.jpeg')

# Display the image
cv2.imshow('Image', image)

# Define a mouse callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Left mouse button clicked at coordinates:', x, y)

# Set the mouse callback function
cv2.setMouseCallback('Image', mouse_callback)

# Wait for a key press
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()