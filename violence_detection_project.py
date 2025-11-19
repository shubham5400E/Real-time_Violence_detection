from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
import cv2
from PIL import Image
import torch
from safetensors.torch import load_file
from transformers import VideoMAEForVideoClassification, VideoMAEConfig, VideoMAEImageProcessor
from supabase import create_client
from datetime import datetime, timedelta
import requests
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import uuid

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch is using device: {device}")

# Flask app setup
app = Flask(__name__)
CORS(app)

SUPABASE_URL = "https://ykaadefarvrzveaqsxho.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlrYWFkZWZhcnZyenZlYXFzeGhvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzYxNjI5MjQsImV4cCI6MjA1MTczODkyNH0.UxyjxOoFfCLkWw2Vyzjg_t2L8BYUtuyDleUrH6EXJYY"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load models
model_weights_path = "D:/violence_detection_project/cpp/model.safetensors"
config_path = "D:/violence_detection_project/cpp/config.json"

video_mae_processor = VideoMAEImageProcessor.from_pretrained(config_path)
video_mae_config = VideoMAEConfig.from_pretrained(config_path)
video_mae_model = VideoMAEForVideoClassification(video_mae_config)

# Load model weights
video_mae_model.load_state_dict(load_file(model_weights_path))
video_mae_model.to(device).eval()

# Global variables
last_notification_time = {}
active_cameras = {}  # Dictionary to store active camera threads: {camera_id: (future, stop_event)}
frame_buffers = {}  # Dictionary to store frame buffers for each camera: {camera_id: []}
video_writers = {}  # Dictionary to store video writers for each camera: {camera_id: cv2.VideoWriter}
violence_detected_flags = {}  # Dictionary to track violence detection state: {camera_id: bool}

# Constants
PRE_BUFFER_SECONDS = 3  # Store 5 seconds of frames before violence detection
POST_BUFFER_SECONDS = 2  # Capture 10 seconds of frames after violence detection
FRAME_RATE = 25  # Assumed frame rate for video capture

# Thread pool for managing camera streams
thread_pool = ThreadPoolExecutor(max_workers=3)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_all_frames(video_path):
    """Extract all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

def store_notification(camera_id, user_id):

    notification_id = str(uuid.uuid4())
    response = supabase.table("notifications").insert({
        "id": notification_id,
        "camera_id": camera_id,
        "user_id": user_id,
        "notification_text": "Violence detected!",
        "video_url": None  # Initialize video_url as None
    }).execute()

    logging.info(f"Notification sent for camera {camera_id}.")
    return notification_id

def update_video_status(video_url, status, result):
    """Update the status of a video analysis in Supabase."""
    supabase.table("video_analysis").update({
        "status": status,
        "results": result
    }).eq("video_url", video_url).execute()

def save_video_clip(camera_id, frames,user_id,notification_id):
    """Save frames as a video clip and upload to Supabase Storage."""
    if not frames:
        return
    print("entered in save video")
    
    if isinstance(frames[0], Image.Image):  # Check if frames are PIL Images
        frames = [np.array(frame) for frame in frames]

    # Create a temporary video file
    output_path = f"temp_violence_clip_{camera_id}.mp4"
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, FRAME_RATE, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()
    video_filename = f"{user_id}/violence_detected_clip/{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    # Upload to Supabase Storage
    with open(output_path, "rb") as f:
        supabase.storage.from_("video-analysis").upload(
            file=f,
            path=f"{user_id}/violence_detected_clip/{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )

    video_url = f"{SUPABASE_URL}/storage/v1/object/public/video-analysis/{video_filename}"

    # Update the notification with the video URL
    supabase.table("notifications").update({
        "video_url": video_url
    }).eq("id", notification_id).execute()
    # Clean up the temporary file
    os.remove(output_path)
    logging.info(f"Video clip saved for camera {camera_id}.")

def process_camera_stream(camera_id, user_id, camera_url, stop_event):
    """Process frames from a camera stream and detect violence."""
    logging.info(f"Starting thread for camera {camera_id}")
    cap = cv2.VideoCapture("http://" + camera_url)
    if not cap.isOpened():
        logging.error(f"Error: Cannot open camera stream {camera_url}")
        return

    previous_frame =[]
    video_frames = [] 
    save_video_frame=[] # Frames to be saved as a video clip
    violence_detected = False
    post_buffer_counter = 0
    notification_sent = False  # Flag to track if a notification has been sent
    SEQUENCE_LENGTH = 16

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Stream ended or error reading frames for camera {camera_id}.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(Image.fromarray(frame_rgb))

            previous_frame.append(video_frames)
            if len(previous_frame) > PRE_BUFFER_SECONDS * FRAME_RATE:
                previous_frame.pop(0) 

            # Process when enough frames are buffered
            if len(video_frames) == SEQUENCE_LENGTH:
                
                # Preprocess frames using VideoMAE processor
                inputs = video_mae_processor(
                    images=video_frames,
                    return_tensors="pt",  # Return PyTorch tensors
                ).to(device)

                # Perform violence detection
                with torch.no_grad():
                    outputs = video_mae_model(**inputs)
                    predicted_class = outputs.logits.argmax(dim=-1).item()

                # Handle violence detection
                if predicted_class == 0:  # Assuming 0 = violence
                    logging.info("Violence detected!")

                    if not violence_detected:
                        # Start saving frames (current frame + next frames)
                        violence_detected = True
                        post_buffer_counter = 0

                        # Send notification only if it hasn't been sent yet
                        if not notification_sent:
                            notification_id = store_notification(camera_id, user_id)
                            notification_sent = True

                    # Extend post-buffering if violence is detected again
                    post_buffer_counter = 0
                else:
                    print("not detected")

                if violence_detected:
                    save_video_frame.extend(previous_frame[:-16])
                    save_video_frame.extend(video_frames)
                    post_buffer_counter += 16
                    print(post_buffer_counter)

                    # Stop capturing after 10 seconds of no violence
                if post_buffer_counter >= POST_BUFFER_SECONDS * FRAME_RATE:
                    #thread_pool.submit(save_video_clip, camera_id, save_video_frame.copy(), user_id, notification_id)
                    #save_video_frame = []
                    violence_detected = False
                    notification_sent = False  # Reset notification flag for the next detection

                # Maintain the sequence length by removing the oldest frame
                video_frames =[]

    except Exception as e:
        logging.error(f"Error processing frames for camera {camera_id}: {e}")
    finally:
        cap.release()
        logging.info(f"Finished processing camera stream for camera {camera_id}.")

def classify_with_video_mae(video_path, group_size=16):
    """Classify a video using the VideoMAE model."""
    frames = extract_all_frames(video_path)
    num_groups = len(frames) // group_size
    predictions = []

    for i in range(num_groups):
        group_frames = frames[i * group_size:(i + 1) * group_size]
        inputs = video_mae_processor(images=group_frames, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = video_mae_model(**inputs)
            predicted_class = outputs.logits.argmax(dim=-1).item()
            predictions.append(predicted_class)

    # Determine overall violence detection based on predictions
    is_violent = 0 in predictions  # Assuming 0 indicates violence
    return is_violent

def process_video(video_url):
    """Process a video for violence detection."""
    response = requests.get(video_url)
    with open("temp_video.mp4", "wb") as f:
        f.write(response.content)

    # Perform violence detection with VideoMAE
    is_violent = classify_with_video_mae("temp_video.mp4")
    result = "Yes" if is_violent else "No"
    logging.info(f"Video processing completed. Result: {result}")
    update_video_status(video_url, "completed", result)

    # Clean up temporary file
    os.remove("temp_video.mp4")

def process_pending_videos():
    """Process all pending videos in Supabase."""
    response = supabase.table("video_analysis").select("*").eq("status", "pending").execute()
    videos = response.data

    for video in videos:
        process_video(video["video_url"])

@app.route('/registerCamera', methods=['POST'])
def add_camera():
    """Register a new camera and start processing its stream."""
    data = request.get_json()
    camera_id = data.get("camera_id")
    user_id = data.get("user_id")
    camera_url = data.get("camera_url")

    if camera_id in active_cameras:
        return jsonify({"message": "Camera is already active."}), 400

    stop_event = threading.Event()
    future = thread_pool.submit(process_camera_stream, camera_id, user_id, camera_url, stop_event)
    active_cameras[camera_id] = (future, stop_event)
    return jsonify({"message": "Camera processing started."})

@app.route('/stopCamera', methods=['POST'])
def stop_camera():
    """Stop processing a camera stream."""
    data = request.get_json()
    camera_id = data.get("camera_id")

    if camera_id not in active_cameras:
        return jsonify({"message": "Camera is not active."}), 404

    future, stop_event = active_cameras[camera_id]
    stop_event.set()  # Signal the thread to stop
    future.result()  # Wait for the thread to finish
    del active_cameras[camera_id]  # Remove the camera from active list
    return jsonify({"message": f"Camera {camera_id} stopped successfully."})

@app.route('/registerVideo', methods=['POST'])
def process_video_request():
    """Process a video for violence detection."""
    data = request.get_json()
    video_url = data.get("video_url")
    logging.info("New video detected for processing.")

    thread_pool.submit(process_video, video_url)
    return jsonify({"message": "Video processing started."})

if __name__ == '__main__':
    # Start processing for all active cameras
    response = supabase.table("cameras").select("*").execute()
    cameras = response.data

    for camera in cameras:
        stop_event = threading.Event()
        future = thread_pool.submit(
            process_camera_stream,
            camera["id"], camera["user_id"], camera["ip_address"], stop_event
        )
        active_cameras[camera["id"]] = (future, stop_event)

    # Start processing pending videos
    thread_pool.submit(process_pending_videos)

    # Run the Flask app
    app.run(debug=False)