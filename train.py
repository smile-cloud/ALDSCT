from ultralytics import YOLO

# Load a model
# model = YOLO("G:\hzh\YOLOv8-Teeth\datasets\yolov8-seg.yaml")  # build a new model from scratch
model = YOLO('yolov8m-pose.pt')  # load a pretrained model (recommended for training)
# model = YOLO('G:\hzh\YOLOv8-Teeth\datasets\yolov8-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Use the model
model.train(data="datasets/coco128-seg.yaml",
            task="pose",
            mode="train",
            workers=0,
            imgsz=640,
            batch=32,
            epochs=50,
            device=0)  # train the model
