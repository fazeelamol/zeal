import cv2
import numpy as np

#web camera
cap = cv2.VideoCapture('video.mp4')
count_line_position =280

#variables-vehicle detection
min_width_rect=80 # width rectangle
min_height_rect=80 #min width rectangle
detect= []
offset = 6#Allowable error between pixel
counter=0
prev_frame_count=0

#Initialize background Subtractor
algo=cv2.createBackgroundSubtractorMOG2()

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

while True:
    ret,frame1 = cap.read()
    if not ret:
        break
    #convert frame to greyscale
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
   # applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    counterShape,h = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    

#draw the blue line on the frame
    cv2.line(frame1,(0,count_line_position),(frame1.shape[1],count_line_position),(255,127,0),3)

#to detect vehicles and draw green squares
    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter=(w>= min_width_rect) and (h>=min_height_rect)
        if not validate_counter:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame1,"vehicle:"+str(counter),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2)



        center =center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)

    #check if vehicle crosses the count line
        for (x,y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position-offset):
                counter+=1
                cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
                detect.remove((x,y))
                print("Vehicle counter:"+str(counter))
    #display vehicle count
    cv2.putText(frame1, "Vehicle count: " + str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#increment vehiles if new vehicles are detected
    if counter >prev_frame_count:
        prev_frame_count = counter

    if counter>28:
        cv2.putText(frame1,"Warning: High traffic volume!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#Display the frame

   # cv2.imshow('Detecter',dilatada)
    cv2.imshow('video original',frame1)
    #cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),5)

    if cv2.waitKey(5) == 13:
        break
cv2.destroyAllWindows()
cap.release()
