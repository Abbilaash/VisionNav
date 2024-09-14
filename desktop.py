import customtkinter as ctk
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import threading
import cv2
import numpy as np
import func
import pyttsx3
import time
import random
import queue

# Initialize the CustomTkinter app
app = ctk.CTk()
app.geometry("900x600")
app.title("VisionNav Results")

titletext = ctk.CTkLabel(master=app, text="VisionNav Dashboard",font=("Arial", 25,"bold"))
titletext.place(x=50, y=35)

# Predictions display on the right side
predictions_textbox = ctk.CTkTextbox(master=app, width=300, height=300)
predictions_textbox.place(x=50, y=200)

# Start detection button on the left
start_button = ctk.CTkButton(master=app, text="Start Object Detection", width=200)
start_button.place(x=50, y=100)

# Pause and Resume buttons
pause_button = ctk.CTkButton(master=app, text="Pause", width=100)
pause_button.place(x=50, y=140)

resume_button = ctk.CTkButton(master=app, text="Resume", width=100)
resume_button.place(x=160, y=140)

# bot conversation box
convo_textbox = ctk.CTkTextbox(master=app, width=500, height=300)
convo_textbox.place(x=400, y=200)

# Flag to control insertion
insertion_paused = False

# List to store predictions
predictions_list = []
predictions_lock = threading.Lock()

voice_engine = pyttsx3.init()
message_queue = queue.Queue()

def tts_worker():
    while True:
        message = message_queue.get()
        voice_engine.say(message)
        voice_engine.runAndWait()
        message_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

# Function to speak consolidated predictions
def speak_predictions():
    while True:
        time.sleep(3)
        with predictions_lock:
            if predictions_list:
                unique_objects = set(predictions_list)

                message_template = random.choice(func.messages())
                message = message_template.format(object_class=unique_objects)

                message_queue.put(message)

                # Insert the message into the conversation textbox
                convo_textbox.configure(state="normal")
                convo_textbox.insert(ctk.END, message + "\n")
                convo_textbox.see(ctk.END)
                convo_textbox.configure(state="disabled")

                # Clear the predictions list
                predictions_list.clear()

# Start the thread to speak predictions
threading.Thread(target=speak_predictions, daemon=True).start()


def pause_insertion():
    global insertion_paused
    insertion_paused = True
def resume_insertion():
    global insertion_paused
    insertion_paused = False

# Custom function to handle predictions and print detected objects
def custom_on_prediction(predictions, frame):
    try:
        frame_hright,frame_width = frame.shape[:2]
        if not insertion_paused:
            for prediction in predictions['predictions']:
                object_class = prediction['class']
                confidence = prediction['confidence']
                # Insert predictions into the textbox
                predictions_textbox.configure(state="normal")
                predictions_textbox.insert(ctk.END, f"Object: {object_class}, Confidence: {confidence:.2f}\n")
                predictions_textbox.see(ctk.END)  # Scroll to the end of the textbox
                predictions_textbox.configure(state="disabled")  # Disable editing
                print(f"Object: {object_class}, Confidence: {confidence:.2f}")

                with predictions_lock:
                    predictions_list.append(object_class)


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

# Function to start detection in a new thread
def start_detection():
    def detection_thread():
        # Initialize a pipeline object
        try:
            pipeline = InferencePipeline.init(
                model_id="living-room-items/1",  # Replace with your Roboflow model ID
                video_reference=0,  # 0 to use the built-in webcam
                api_key="G0HdUVNRRR2g5eGLP20Z",  # Your Roboflow API key
                on_prediction=custom_on_prediction,  # Function to handle predictions
            )
            pipeline.start()
            pipeline.join()

        except Exception as e:
            print(f"Error initializing pipeline: {e}")
        finally:
            cv2.destroyAllWindows()

    # Start the detection thread
    threading.Thread(target=detection_thread, daemon=True).start()

# Link the button to the start detection function
start_button.configure(command=start_detection)
pause_button.configure(command=pause_insertion)
resume_button.configure(command=resume_insertion)

# Start the Tkinter main loop
app.mainloop()
