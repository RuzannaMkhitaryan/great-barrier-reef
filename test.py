from ultralytics import YOLO
model = YOLO("models/yolo11s_v3_best_mAP50-0.619.pt")
print(model.task)  # should print "detect"