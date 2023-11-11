import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt

model_type = "DPT_Hybrid"

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920,  1080))

while(cap.isOpened()): 
    ret, frame = cap.read()  
    
    if ret:
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(frame).to(device)
        
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        output = prediction.cpu().numpy()
            
        cv2.imshow("output", output/2048)
        cv2.imshow("frame", frame)
        
        out.write(frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    
cap.release()
out.release
  
# De-allocate any associated memory usage  
cv2.destroyAllWindows() 
