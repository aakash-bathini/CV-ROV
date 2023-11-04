from transformers import pipeline
from PIL import Image
import cv2 
import numpy as np

checkpoint = "vinvino02/glpn-nyu"
depth_estimator = pipeline("depth-estimation", model=checkpoint)

cap = cv2.VideoCapture(0)

while(True): 
    ret, frame = cap.read()  
    
    if ret:
        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        pil_image = Image.fromarray(color_coverted)
        pil_image.thumbnail((500, 500))
        
        predictions = depth_estimator(pil_image)
        depth = np.array(predictions["depth"])
        
        cv2.imshow('depth', depth)  
        cv2.imshow('frame', frame)  
        
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    
cap.release()
  
# De-allocate any associated memory usage  
cv2.destroyAllWindows() 
