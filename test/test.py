# Import necessary modules
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import cv2
import numpy as np
import supervision as sv

# TASKS:
# make the app to detect appropriate objects (center of the frame and adjacent obj)
# make the app to detect vehicles
# make the app to find the speed of vehicles
# make the app faster

def get_box_center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

# Custom function to handle predictions and print detected objects
def custom_on_prediction(predictions, frame):
    try:
        frame_img = frame.image    #converting frame to image to find the height and width
        frame_height, frame_width = frame_img.shape[:2]
        frame_center_x, frame_center_y = frame_width // 2, frame_height // 2

        min_distance = float('inf')
        center_object = None

        for prediction in predictions['predictions']:
            # object_class = prediction['class']
            # confidence = prediction['confidence']

            bbox_x = prediction['x']
            bbox_y = prediction['y']
            bbox_width = prediction['width']
            bbox_height = prediction['height']

            obj_center_x = bbox_x + bbox_width // 2
            obj_center_y = bbox_y + bbox_height // 2
            distance_to_center = ((obj_center_x - frame_center_x) ** 2 + (obj_center_y - frame_center_y) ** 2) ** 0.5

            if distance_to_center < min_distance:
                min_distance = distance_to_center
                center_object = prediction

            # print(f"Object: {object_class}, Confidence: {confidence:.2f}")
            # center_object is the object at the center of the frame
            names = ['Book', 'Chair', 'Clock', 'Cupboard', 'Curtain', 'Frame', 'Lamp', 'Rug', 'Socket', 'Switch', 'Table', 'Television', 'Vase', 'window']

            print("Cnter onject is: ",center_object)


        # Render the bounding boxes on the frame
        render_boxes(predictions, frame)

        # Check if the frame is valid and display it (if needed)
        if frame is not None and isinstance(frame, np.ndarray):
            cv2.imshow('Frame', frame)

        # Press 'q' to exit the detection
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    except Exception as e:
        # print(f"Error in custom_on_prediction: {e}")
        pass

vehivle_trackers = []

# Initialize a pipeline object
try:
    pipeline = InferencePipeline.init(
        model_id="visionnav-xkpmk/1",
        video_reference=0,
        api_key="G0HdUVNRRR2g5eGLP20Z",
        on_prediction=custom_on_prediction,
    )

    pipeline.start()
    pipeline.join()
except Exception as e:
    print(f"Error initializing pipeline: {e}")
finally:
    cv2.destroyAllWindows()
