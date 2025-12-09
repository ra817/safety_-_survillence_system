from ultralytics import YOLO
import yaml
import os

# ----------------------------------------------------------------------
# 1. PATHS
# ----------------------------------------------------------------------
train_images = "data/human_detection/images/train"   
val_images   = "data/human_detection/images/val"     

train_labels = train_images.replace("images", "labels")
val_labels   = val_images.replace("images", "labels")

assert os.path.exists(train_images), f"Train images not found: {train_images}"
assert os.path.exists(val_images), f"Val images not found: {val_images}"
assert os.path.exists(train_labels), f"Train labels not found: {train_labels}"
assert os.path.exists(val_labels), f"Val labels not found: {val_labels}"

# ----------------------------------------------------------------------
# 2. CREATE data.yaml AUTOMATICALLY
# ----------------------------------------------------------------------
data_yaml = {
    "path": ".",
    "train": train_images,
    "val": val_images,
    "names": {0: "person"}
}

with open("human_eval.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("\nGenerated human_eval.yaml:")
print(yaml.dump(data_yaml))

# ----------------------------------------------------------------------
# 3. LOAD YOUR TRAINED MODEL
# ----------------------------------------------------------------------
model = YOLO("models/YOLO12/yolo12n.pt")   

# ----------------------------------------------------------------------
# 4. EVALUATION SETTINGS (YOU CAN CHANGE THESE)
# ----------------------------------------------------------------------
imgsz = 640         # evaluation input size
conf  = 0.25        # confidence threshold for filtering predictions
nms_iou = 0.7       # IoU threshold for NMS (non-max suppression)

# ----------------------------------------------------------------------
# 5. EVALUATE TRAIN SET
# ----------------------------------------------------------------------
print("\n=== Evaluating TRAIN SET with Ground Truth ===")

results_train = model.val(
    data="human_eval.yaml",
    split="train",
    imgsz=imgsz,
    conf=conf,
    iou=nms_iou
)

print("\n[TRAIN METRICS]")
print("Precision :", results_train.box.mp)
print("Recall    :", results_train.box.mr)
print("mAP50     :", results_train.box.map50)
print("mAP50-95  :", results_train.box.map)

# ----------------------------------------------------------------------
# 6. EVALUATE VAL SET
# ----------------------------------------------------------------------
print("\n=== Evaluating VAL SET with Ground Truth ===")

results_val = model.val(
    data="human_eval.yaml",
    split="val",
    imgsz=imgsz,
    conf=conf,
    iou=nms_iou
)

print("\n[VAL METRICS]")
print("Precision :", results_val.box.mp)
print("Recall    :", results_val.box.mr)
print("mAP50     :", results_val.box.map50)
print("mAP50-95  :", results_val.box.map)
