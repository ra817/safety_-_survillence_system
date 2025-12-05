from ultralytics import YOLO
model = YOLO("models/YOLO12/yolo12s.pt")
results = model("data/3.png", save=True)
