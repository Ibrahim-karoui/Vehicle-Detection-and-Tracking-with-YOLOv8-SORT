import os
import sys
from ultralytics import YOLO
import numpy as np
import cv2


#Use `BASE_DIR` to make the project portable.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_video_path = os.path.join(BASE_DIR, "data\\2103099-uhd_3840_2160_30fps.mp4")
output_video_path = os.path.join(BASE_DIR, "runs\\results.avi")

#charge the yolo model
model = YOLO('yolov8n')


#charge the video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: can't open the video !")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#create output video
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width,height))

#initialise sort
sys.path.append(BASE_DIR)
from sort.sort import *
tracker = Sort()

#define supported vehicules
vehicules = [2, 3, 5, 7]

ret = True
while True :
    ret, frame = cap.read()
    if not ret :
        break
    detections = model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist() :
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicules :
            detections_.append([x1, y1, x2, y2, score])
    if len(detections_ )> 0 :
        detections_ = np.array(detections_)
    else :
        detections_ = np.empty((0,5))
        
    tracks = tracker.update(detections_)
    for  track in tracks :
        x1, y1, x2, y2, track_id = track.astype(int)
        #display bounding boxes
        cv2.rectangle(frame, (x1, y1), (x2,y2), [0,255,0], 3)
        #display car ids
        cv2.putText(frame, f"ID {int(track_id)}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)

print("Finished successfully! You can find the output video in :", output_video_path)
cap.release()
out.release()
cv2.destroyAllWindows()
        





