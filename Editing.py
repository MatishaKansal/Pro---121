import numpy as np
import time 
import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
image = cv2.VideoWriter('Output.avi', fourcc, 20.0, (640, 480))

bg = 0
for i in range(60):
    frame, bg = cap.read()

frame = cv2.resize(frame, (640, 480))

bg = np.flip(bg, axis=1)

while(cap.isOpened()):
    ret,img = cap.read()
    if not ret:
        break
    img = np.flip(img, axis=1)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    
    u_black = np.array([104, 153, 70])
    l_black = np.array([30,30,0])
    mask = cv2.inRange(hsv, u_black, l_black)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    res_1 = cv2.bitwise_and(frame, frame, mask=mask)

    res_2 = cv2.bitwise_and(bg,bg,mask = mask)
 
  
    f = frame - res_1
    f = np.where(f== 0 , image, f)

    final_output = cv2.addWeighted(res_1,1.5, res_2, 1.5,0)
    image.write(final_output)
    cv2.imshow('magic', final_output)
    cv2.waitKey(1)


cap.release()
out.release()
cv2.destroyAllWindows()
