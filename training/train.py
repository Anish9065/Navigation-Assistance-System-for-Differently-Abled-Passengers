import os
from ultralytics import YOLO

def train_model():
    """
    Trains the YOLOv8n model on the custom dataset.
    """
    # Load a pre-trained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    data_yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'data.yaml'))

    print(f"Starting training with dataset configuration: {data_yaml_path}")

    # Train the model
    # The ultralytics library handles automatic saving of the best model to 'runs/detect/train/weights/best.pt'
    # It also handles logging, evaluation metrics (precision, recall, mAP) and plotting confusion matrix automatically.
    # We pass augmentation parameters to the train function.
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        project="outputs",
        name="navigation_assist_model",
        exist_ok=True,
        # Augmentation pipeline parameters
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        save=True # automatically saves best model
    )

    print("Training completed. Results saved in 'outputs/navigation_assist_model'.")

    # Evaluate model on validation set
    metrics = model.val()
    print("Validation metrics:")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")

if __name__ == "__main__":
    train_model()
