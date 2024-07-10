from ultralytics import YOLO
if __name__ == '__main__':

    # Load a model
    model = YOLO('yolov8s.yaml')  # build a new model from YAML
    model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r'E:\HB_WuChang\code\dc\ultralytics-main\ultralytics\cfg\datasets\rizline2.yaml', epochs=32, imgsz=640)
