import cv2
import time
import numpy as np
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes are in format [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

class NavigationDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.5, iou_thresh=0.45):
        """
        Initializes the YOLOv8 detector with ByteTrack integration.
        """
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.class_names = {
            0: "normal_person",
            1: "wheelchair_user",
            2: "blind_person",
            3: "crutch_user"
        }

        # Colors for drawing bounding boxes (BGR format)
        self.colors = {
            0: (0, 255, 0),     # Green for normal person
            1: (0, 0, 255),     # Red for wheelchair
            2: (255, 0, 0),     # Blue for blind person
            3: (0, 255, 255)    # Yellow for crutch user
        }

    def process_frame(self, frame):
        """
        Runs object detection and tracking on a single frame.
        Applies logic to handle overlapping classes (e.g., person + wheelchair).
        """
        # Run YOLOv8 tracking, using ByteTrack tracker by default
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detected_objects = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            # Pre-processing logic: Reduce false positives
            # If normal person and wheelchair are highly overlapping, only report wheelchair user

            # 1. Group indices by class
            person_indices = [i for i, c in enumerate(clss) if c == 0]
            wheelchair_indices = [i for i, c in enumerate(clss) if c == 1]

            # Track which persons to suppress
            suppressed_persons = set()

            for p_idx in person_indices:
                p_box = boxes[p_idx]
                for w_idx in wheelchair_indices:
                    w_box = boxes[w_idx]

                    iou = calculate_iou(p_box, w_box)
                    # If high overlap, suppress the normal_person detection
                    if iou > 0.3:
                        suppressed_persons.add(p_idx)
                        break

            # Draw and append
            for i, (box, cls, conf, track_id) in enumerate(zip(boxes, clss, confs, track_ids)):
                if i in suppressed_persons:
                    continue

                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)

                # Default draw
                color = self.colors.get(class_id, (255, 255, 255))
                label = f"ID: {track_id} {self.class_names.get(class_id, 'Unknown')} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                detected_objects.append({
                    'id': track_id,
                    'class': self.class_names.get(class_id, "Unknown"),
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })

        return frame, detected_objects

    def run_video_stream(self, source=0):
        """
        Runs the detection on a video stream or webcam.
        """
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Error: Could not open video source {source}.")
            return

        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.success() if hasattr(cap, 'success') else cap.read()
            if not ret:
                break

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            processed_frame, detections = self.process_frame(frame)

            # Draw FPS
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Navigation Assistant Stream", processed_frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test initialization
    detector = NavigationDetector()
    print("Navigation Detector initialized successfully.")
