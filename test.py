import torch

model = torch.load("models/YOLO12/yolo12n.pt", map_location="cpu")["model"]
print(model)