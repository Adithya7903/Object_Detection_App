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
from pushbullet import Pushbullet
import time  # Import the time module

# Pushbullet API key
PUSHBULLET_API_KEY = 'o.q7S5u10icooJdJvzxmBvY2GbuW1fNyeI'
pb = Pushbullet(PUSHBULLET_API_KEY)

# Global variables for OpenCV-related objects and flags
cap = None
is_camera_on = False
frame_count = 0
frame_skip_threshold = 3
model = YOLO('yolov8s.pt')
video_paused = False
restricted_area_enabled = False
restricted_area_pts = []

# Global variable for the CSV file name
log_file = "detection_log.csv"
restricted_area_file = "restricted_area.txt"

# Global variable to store the last time a notification was sent or a detection was logged
last_notification_time = 0
notification_cooldown = 5  # Set the notification cooldown to 5 seconds

last_log_time = 0
log_cooldown = 5  # Set the log cooldown to 5 seconds

# Function to read classes from file
def read_classes_from_file(file_path):
    with open(file_path, 'r') as file:
        classes = [line.strip() for line in file]
    return classes

# Function to log detected items to a CSV file with a cooldown
def log_detection(detection_time, class_name):
    global last_log_time
    current_time = time.time()
    
    # Check if cooldown time has passed since the last detection log
    if current_time - last_log_time >= log_cooldown:
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([detection_time, class_name])
        
        print(f"Detection logged: {class_name} at {detection_time}")
        # Update the last log time
        last_log_time = current_time

# Function to send a Pushbullet notification with a cooldown
def send_push_notification(title, body):
    global last_notification_time
    current_time = time.time()
    
    # Check if cooldown time has passed since the last notification
    if current_time - last_notification_time >= notification_cooldown:
        try:
            push = pb.push_note(title, body)
            print(f"Notification sent: {push}")
            # Update the last notification time
            last_notification_time = current_time
        except Exception as e:
            print(f"Error sending notification: {e}")

# Function to initialize the webcam
def initialize_webcam():
    global cap, is_camera_on, video_paused
    if not is_camera_on:
        cap = cv2.VideoCapture(0)  # Use the default webcam (you can change the index if needed)
        is_camera_on = True
        video_paused = False
        update_canvas()  # Start updating the canvas

# Function to stop the webcam feed
def stop_webcam():
    global cap, is_camera_on, video_paused
    if cap is not None:
        cap.release()
        is_camera_on = False
        video_paused = False
        update_status_label("Webcam Stopped")

# Function to pause or resume the video
def pause_resume_video():
    global video_paused
    video_paused = not video_paused
    update_status_label("Video Paused" if video_paused else "Video Resumed")

# Function to start video playback from a file
def select_file():
    global cap, is_camera_on, video_paused
    if is_camera_on:
        stop_webcam()  # Stop the webcam feed if running
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        is_camera_on = True
        video_paused = False
        update_canvas()  # Start updating the canvas with the video
        update_status_label(f"Video Selected: {file_path}")

# Function to handle mouse events for creating a restricted area
def handle_mouse(event, x, y, flags, param):
    global restricted_area_pts
    if restricted_area_enabled:
        if event == cv2.EVENT_LBUTTONDOWN:
            restricted_area_pts = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            restricted_area_pts.append((x, y))
            cv2.rectangle(frame, restricted_area_pts[0], restricted_area_pts[1], (255, 0, 0), 2)
            cv2.imshow('Canvas', frame)

# Function to create a window with the first frame for drawing a restricted area
def create_restricted_area():
    global restricted_area_enabled, cap, frame
    restricted_area_enabled = not restricted_area_enabled

    if restricted_area_enabled and cap is not None:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1020, 500))

        cv2.namedWindow('Canvas')
        cv2.setMouseCallback('Canvas', handle_mouse)

        # Display the first frame in the window
        cv2.imshow('Canvas', frame)

# Function to save restricted area coordinates
def save_restricted_area():
    global restricted_area_pts
    if restricted_area_pts:
        with open(restricted_area_file, 'w') as file:
            for point in restricted_area_pts:
                file.write(f"{point[0]},{point[1]}\n")
        update_status_label("Restricted Area Saved")

# Function to load restricted area coordinates
def load_restricted_area():
    global restricted_area_pts
    restricted_area_pts = []
    try:
        with open(restricted_area_file, 'r') as file:
            for line in file:
                x, y = map(int, line.strip().split(','))
                restricted_area_pts.append((x, y))
        update_status_label("Restricted Area Loaded")
    except FileNotFoundError:
        update_status_label("No saved restricted area.")

# Function to update the Canvas with the webcam frame or video frame
def update_canvas():
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

                if restricted_area_enabled:
                    cv2.rectangle(frame, restricted_area_pts[0], restricted_area_pts[1], (255, 0, 0), 2)

                results = model.predict(frame)
                a = results[0].boxes.data
                px = pd.DataFrame(a).astype("float")
                for index, row in px.iterrows():
                    x1, y1, x2, y2, _, d = map(int, row)
                    c = class_list[d]
                    if selected_class == "All" or c == selected_class:
                        if not restricted_area_enabled or (
                                restricted_area_enabled and is_point_in_rect((x1, y1), restricted_area_pts)
                                and is_point_in_rect((x2, y2), restricted_area_pts)):
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                            # Log the detection
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            log_detection(current_time, c)

                            # Send Pushbullet notification for each detection
                            notification_title = f"Object Detected: {c}"
                            notification_body = f"At {current_time}, {c} was detected."
                            send_push_notification(notification_title, notification_body)

                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                canvas.img = photo
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        canvas.after(10, update_canvas)

# Function to check if a point is inside a rectangle
def is_point_in_rect(point, rect_pts):
    x, y = point
    x1, y1 = rect_pts[0]
    x2, y2 = rect_pts[1]
    return x1 < x < x2 and y1 < y < y2

# Function to update the status label
def update_status_label(message):
    status_label.config(text=message)

# Function to quit the application
def quit_app():
    stop_webcam()
    cv2.destroyAllWindows()
    root.quit()
    root.destroy()

# Create the main Tkinter window
root = tk.Tk()
root.title("Object Detection App")

# Create a Canvas widget to display the webcam feed or video
canvas = tk.Canvas(root, width=1020, height=500)
canvas.pack(fill='both', expand=True)

# Create a status label
status_label = tk.Label(root, text="Welcome to Object Detection App", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(side=tk.BOTTOM, fill=tk.X)

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

# Button to create a restricted area
restricted_area_button = tk.Button(button_frame, text="Create Restricted Area", command=create_restricted_area)
restricted_area_button.pack(side='left')

# Button to save restricted area
save_button = tk.Button(button_frame, text="Save Restricted Area", command=save_restricted_area)
save_button.pack(side='left')

# Button to load restricted area
load_button = tk.Button(button_frame, text="Load Restricted Area", command=load_restricted_area)
load_button.pack(side='left')

quit_button = tk.Button(button_frame, text="Quit", command=quit_app)
quit_button.pack(side='left')

initial_image = Image.open('1st.jpg')  # Replace 'initial_image.jpg' with your image path
initial_photo = ImageTk.PhotoImage(image=initial_image)
canvas.img = initial_photo
canvas.create_image(0, 0, anchor=tk.NW, image=initial_photo)

root.mainloop()
