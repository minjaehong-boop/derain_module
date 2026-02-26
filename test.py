import cv2
from ultralytics import YOLO
from derain_tool import deraining

model = YOLO("./model/detect/yolov8m.pt")

input_dir = ["./input/input.mp4"]



for src in input_dir:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"Skip (cannot open): {src}")
        continue

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        frame = deraining(frame)
        results = model(frame)
        cv2.imshow("YOLOv8 Inference", results[0].plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
