import win32event
import win32api
import sys
from winerror import ERROR_ALREADY_EXISTS
import tkinter as tk
from tkinter.ttk import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import threading
import csv
from datetime import datetime

# Global variables for OpenCV-related objects and flags
cap = None
is_camera_on = False
frame_count = 0
frame_skip_threshold = 3
model = YOLO('yolov8s.pt')
video_paused = False

# Global variable for the CSV file name
log_file = "detection_log.csv"

# Function to read classes from file
def read_classes_from_file(file_path):
    """
    Reads classes from the specified file and returns a list.
    """
    with open(file_path, 'r') as file:
        classes = [line.strip() for line in file]
    return classes

# Function to log detected items to a CSV file
def log_detection(detection_time, class_name):
    """
    Logs detected items to a CSV file.
    """
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([detection_time, class_name])

# Function to initialize the webcam
def initialize_webcam():
    """
    Initializes the webcam and sets up necessary variables.
    """
    global cap, is_camera_on, video_paused
    if not is_camera_on:
        cap = cv2.VideoCapture(0)  # Use the default webcam (you can change the index if needed)
        is_camera_on = True
        video_paused = False
        update_canvas()  # Start updating the canvas

# Function to stop the webcam feed
def stop_webcam():
    """
    Stops the webcam feed and releases resources.
    """
    global cap, is_camera_on, video_paused
    if cap is not None:
        cap.release()
        is_camera_on = False
        video_paused = False

# Function to pause or resume the video
def pause_resume_video():
    """
    Pauses or resumes the video feed.
    """
    global video_paused
    video_paused = not video_paused

# Function to start video playback from a file
def select_file():
    """
    Opens a file dialog to select a video file for playback.
    """
    global cap, is_camera_on, video_paused
    if is_camera_on:
        stop_webcam()  # Stop the webcam feed if running
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        is_camera_on = True
        video_paused = False
        update_canvas()  # Start updating the canvas with the video

# Function to update the Canvas with the webcam frame or video frame
def update_canvas():
    """
    Updates the Canvas with the webcam frame or video frame.
    """
    global is_camera_on, frame_count, video_paused
    if is_camera_on:
        if not video_paused:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                if frame_count % frame_skip_threshold != 0:
                    canvas.after(10, update_canvas)
                    return

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (1020, 500))
                selected_class = class_selection.get()

                results = model.predict(frame)
                a = results[0].boxes.data
                px = pd.DataFrame(a).astype("float")
                for index, row in px.iterrows():
                    x1, y1, x2, y2, _, d = map(int, row)
                    c = class_list[d]
                    if selected_class == "All" or c == selected_class:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                        # Log the detection
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_detection(current_time, c)

                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                canvas.img = photo
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        canvas.after(10, update_canvas)

# Function to quit the application
def quit_app():
    """
    Quits the application and releases resources.
    """
    stop_webcam()
    root.quit()
    root.destroy()

# Create the main Tkinter window
root = tk.Tk()
root.title("YOLO v8 My App")

# Create a Canvas widget to display the webcam feed or video
canvas = tk.Canvas(root, width=1020, height=500)
canvas.pack(fill='both', expand=True)

class_list = read_classes_from_file('coco.txt')

class_selection = tk.StringVar()
class_selection.set("All")  # Default selection is "All"
class_selection_label = tk.Label(root, text="Select Class:")
class_selection_label.pack(side='left')
class_selection_entry = tk.OptionMenu(root, class_selection, "All", *class_list)
class_selection_entry.pack(side='left')

# Create a frame to hold the buttons
button_frame = tk.Frame(root)
button_frame.pack(fill='x')

# Create descriptive buttons
play_button = tk.Button(button_frame, text="Start Webcam", command=initialize_webcam)
play_button.pack(side='left')

stop_button = tk.Button(button_frame, text="Stop Webcam", command=stop_webcam)
stop_button.pack(side='left')

file_button = tk.Button(button_frame, text="Select Video File", command=select_file)
file_button.pack(side='left')

pause_button = tk.Button(button_frame, text="Pause/Resume", command=pause_resume_video)
pause_button.pack(side='left')

quit_button = tk.Button(button_frame, text="Quit", command=quit_app)
quit_button.pack(side='left')

initial_image = Image.open('1st.jpg')  # Replace 'initial_image.jpg' with your image path
initial_photo = ImageTk.PhotoImage(image=initial_image)
canvas.img = initial_photo
canvas.create_image(0, 0, anchor=tk.NW, image=initial_photo)

root.mainloop()
