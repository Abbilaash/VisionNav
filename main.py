from inference import InferencePipeline
import cv2
import numpy as np
from collections import deque
import time

# Constants
DETECTION_INTERVAL = 1  # seconds between speed calculations
MAX_TRACKING_DISTANCE = 100  # maximum pixel distance for tracking
SPEED_ESTIMATION_POINTS = 5  # number of points to use for speed estimation
PIXELS_PER_METER = 10  # needs calibration based on your camera setup

class VehicleTracker:
    def __init__(self, box, class_name):
        self.boxes = deque(maxlen=SPEED_ESTIMATION_POINTS)
        self.timestamps = deque(maxlen=SPEED_ESTIMATION_POINTS)
        self.class_name = class_name
        self.update(box)
    
    def update(self, box):
        self.boxes.append(box)
        self.timestamps.append(time.time())
    
    def estimate_speed(self):
        if len(self.boxes) < 2:
            return None
        
        pixel_distance = np.linalg.norm(
            np.array(self.boxes[-1][:2]) - np.array(self.boxes[0][:2])
        )
        time_elapsed = self.timestamps[-1] - self.timestamps[0]
        
        if time_elapsed == 0:
            return None
        
        speed = (pixel_distance / PIXELS_PER_METER) / time_elapsed * 3.6  # km/h
        return speed

def get_box_center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

vehicle_trackers = []

def custom_on_prediction(predictions, frame):
    try:
        frame_img = frame.image.copy()
        current_time = time.time()
        
        # Process detections
        for prediction in predictions['predictions']:
            class_name = prediction['class']
            box = [prediction['x'], prediction['y'], 
                   prediction['x'] + prediction['width'], 
                   prediction['y'] + prediction['height']]
            
            # Draw bounding box for all objects
            cv2.rectangle(frame_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame_img, class_name, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Process vehicles
            if class_name.lower() in ['car', 'truck', 'bus', 'motorcycle']:
                vehicle_center = get_box_center(box)
                
                # Find the closest tracker
                closest_tracker = None
                min_distance = float('inf')
                for tracker in vehicle_trackers:
                    if tracker.class_name == class_name:
                        distance = np.linalg.norm(np.array(vehicle_center) - np.array(get_box_center(tracker.boxes[-1])))
                        if distance < min_distance and distance < MAX_TRACKING_DISTANCE:
                            min_distance = distance
                            closest_tracker = tracker
                
                if closest_tracker:
                    closest_tracker.update(box)
                else:
                    vehicle_trackers.append(VehicleTracker(box, class_name))
        
        # Estimate and display speeds
        for tracker in vehicle_trackers:
            if current_time - tracker.timestamps[0] >= DETECTION_INTERVAL:
                speed = tracker.estimate_speed()
                if speed is not None:
                    center = get_box_center(tracker.boxes[-1])
                    cv2.putText(frame_img, f"{speed:.1f} km/h", 
                                (center[0] - 30, center[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Frame', frame_img)
        
        # Remove old trackers
        vehicle_trackers[:] = [t for t in vehicle_trackers if current_time - t.timestamps[-1] < DETECTION_INTERVAL]
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    except Exception as e:
        print(f"Error in custom_on_prediction: {e}")

# Initialize pipeline
try:
    pipeline = InferencePipeline.init(
        model_id="living-room-items/1",  # Replace with your actual model ID
        video_reference=1,  # Adjust if needed for your camera setup
        api_key="G0HdUVNRRR2g5eGLP20Z",  # Replace with your actual API key
        on_prediction=custom_on_prediction,
    )

    pipeline.start()
    pipeline.join()
except Exception as e:
    print(f"Error initializing or running pipeline: {e}")
finally:
    cv2.destroyAllWindows()