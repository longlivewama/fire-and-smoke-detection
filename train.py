
## 4. Convert Notebook to Python Scripts

Create these scripts from your notebook code:

### train.py
```python
from roboflow import Roboflow
from ultralytics import YOLO

def train_model():
    # Initialize Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("yang-nan").project("fire-and-smoke-detection-yqrpi")
    dataset = project.version(12).download("yolov8")
    
    # Train YOLOv8 model
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=35,
        imgsz=832,
        optimizer="SGD",
        lr0=0.001,
        patience=10
    )

if __name__ == "__main__":
    train_model()
