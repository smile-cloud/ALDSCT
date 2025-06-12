# export_onnx.py

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('runs/segment/train/best.pt')  # load a custom trained

# Export the model
model.export(format='onnx')