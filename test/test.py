# Import necessary modules
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import cv2
import numpy as np


# Custom function to handle predictions and print detected objects
def custom_on_prediction(predictions, frame):
    try:
        frame = np.array(frame)
        # frame_hright,frame_width = frame.shape[:2]
        for prediction in predictions['predictions']:
            object_class = prediction['class']
            confidence = prediction['confidence']
            print(f"Object: {object_class}, Confidence: {confidence:.2f}")

        # Render the bounding boxes on the frame
        render_boxes(predictions, frame)

        # Check if the frame is valid and display it (if needed)
        if frame is not None and isinstance(frame, np.ndarray):
            cv2.imshow('Predictions', frame)

        # Press 'q' to exit the detection
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    except Exception as e:
        print(f"Error in custom_on_prediction: {e}")

# Initialize a pipeline object
try:
    pipeline = InferencePipeline.init(
        model_id="living-room-items/1",
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
