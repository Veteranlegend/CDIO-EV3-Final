from ultralytics import YOLO

# Load the YOLO model
model = YOLO("/Users/ahmadhaj/Desktop/CDIOProjekt/virtualenv/Server/Assets/test6.pt")


# Export the model
model.export(format='openvino', imgsz=640)

