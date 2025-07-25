from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import os

crack_model = YOLO("C:\Programming\crack_detection\models\crack_segmentation_model_02.pt")      
spalling_model = YOLO("C:\Programming\crack_detection\models\spalling_segmentation_model_01.pt")
video_path = r"C:\Programming\crack_detection\test_vid.mp4"
image_folder = "video_frames"
json_folder = "detection_json"
desired_fps = 5
batch_size = 10

def process_video_to_frames(video_path, image_folder, desired_fps):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error, could not open video file")
        return
    
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_capture_interval = np.ceil(video_fps / desired_fps)
    print(f"Video fps: {video_fps:.2f}")

    frame_count = 0
    saved_frame_count = 0

    frames = []

    while True:
        success, frame = video.read()

        if not success:
            break

        if frame_count % frame_capture_interval == 0:
            saved_frame_count += 1

            filename = f"frame_{saved_frame_count:06d}.jpg"
            output_path = os.path.join(image_folder, filename)

            cv2.imwrite(output_path, frame)
            frames.append(output_path)

        frame_count += 1

    video.release()
    print(f"Successfully extracted {saved_frame_count} frames from the video")
    return frames

def process_images_to_json(crack_model, spalling_model, image_paths, json_folder, batch_size):
    for i in range(0, len(image_paths) + batch_size, batch_size):
        batch_paths = image_paths[i:min(i+batch_size, len(image_paths))]

        try:
            crack_results = crack_model(batch_paths, verbose=False)
            spall_results = spalling_model(batch_paths, verbose=False)
        except Exception as e:
            print(f"Error processing {image_paths}: {e}")
            return

        for crack_result, spall_result, image_path in zip(crack_results, spall_results, batch_paths):
            image_filename = os.basename(image_path)

            output_data = {
                "filename": image_filename,
                "crack_detections": [],
                "spall_detections": []
            }

            class_names = result.names

            


if (__name__ == "__main__"):
    image_paths = process_video_to_frames(video_path, image_folder, desired_fps)
    #process_images_to_json(crack_model, spalling_model, image_paths, json_folder, batch_size)