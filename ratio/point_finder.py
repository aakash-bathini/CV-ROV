import cv2
import numpy as np

# Import the image
cap = cv2.VideoCapture(0)

def undistort(img):
    ret = 0.9610454141475719
    mtx = np.array([
        [929.02095583, 0.00000000e+00, 978.01846836],
        [0.00000000e+00, 927.4997145, 6.68826192e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    dist = np.array([[-0.3821828, 0.17840163, 0.00063669, 0.00090362, -0.04303795]])

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    x = x + 50
    y = y + 50
    w = w - 50
    h = h - 50
    dst = dst[y:y+h, x:x+w]
    
    return dst

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        frame = undistort(frame)
        if cv2.waitKey(1) & 0xFF == ord('s'): 
            image = frame
            break
            
        cv2.imshow("Capture", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Display the image
cv2.imshow('Image', image)
cv2.imwrite("image.jpeg", image)

# Define a mouse callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image,(x,y),25,(255,0,0),-1)
        print('Left mouse button clicked at coordinates:', x, y)
        cv2.imshow('Image', image)
        

# Set the mouse callback function
cv2.setMouseCallback('Image', mouse_callback)

# Wait for a key press
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()