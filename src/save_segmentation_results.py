'''
This code is to process a video and turn it into a 3d reconstruction.
First step is to run the first function with the video you want to use.
Then run the next function to process the images into json files.
Then use those same images in Agisoft Metashape to create the 3d model of the structure.
Make sure not to run both at the same time since they both use a lot of GPU resources.
Once you have the 3d model and the json files, run the final function to combine them.
'''

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import os

crack_model = YOLO("C:\Programming\crack_detection\models\crack_segmentation_model_02.pt")      
spalling_model = YOLO("C:\Programming\crack_detection\models\spalling_segmentation_model_01.pt")
video_path = r"C:\Programming\crack_detection\videos\vid1.MP4"
image_folder = r"videos\vid1_frames"
json_folder = r"videos\vid1_json"
desired_fps = 1
batch_size = 10

def convert_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def process_video_to_frames(video_path, image_folder, desired_fps):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error, could not open video file")
        return
    
    os.makedirs(image_folder, exist_ok=False) 
    #If folder exists should throw exception, don't want to potentially delete files or have them in the same folder

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

        frame_count += 1

    video.release()
    print(f"Successfully extracted {saved_frame_count} frames from the video")

def process_images_to_json(crack_model, spalling_model, image_folder, json_folder, batch_size):
    os.makedirs(json_folder, exist_ok=True)
    try:
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"No images found in '{image_folder}'. Please check the path.")
            return
    except FileNotFoundError:
        print(f"Error: The directory '{image_folder}' was not found.")
        return
    
    for i in range(0, len(image_files), batch_size):
        batch_filenames = image_files[i:i+batch_size]
        batch_paths = [os.path.join(image_folder, f) for f in batch_filenames]

        try:
            crack_results = crack_model(batch_paths, verbose=False)
            spall_results = spalling_model(batch_paths, verbose=False)
        except Exception as e:
            print(f"Error processing {batch_paths}: {e}")
            return

        for crack_result, spall_result, image_filename in zip(crack_results, spall_results, batch_filenames):

            output_data = {
                "filename": image_filename,
                "crack_detections": [],
                "spall_detections": []
            }

            if crack_result.boxes is not None:
                for j in range(len(crack_result.boxes)):
                    box = crack_result.boxes[j]

                    confidence = float(box.conf[0])
                    
                    detection_info = {
                        "confidence": confidence,
                        "bounding_box": box.xywhn[0].tolist(),
                    }

                    if crack_result.masks is not None and crack_result.masks.xy:
                        detection_info["segmentation_mask"] = crack_result.masks.xy[j].tolist()
                    
                    output_data["crack_detections"].append(detection_info)

            if spall_result.boxes is not None:
                for j in range(len(spall_result.boxes)):
                    box = spall_result.boxes[j]

                    confidence = float(box.conf[0])
                    
                    detection_info = {
                        "confidence": confidence,
                        "bounding_box": box.xywhn[0].tolist(),
                    }

                    if spall_result.masks is not None and spall_result.masks.xy:
                        detection_info["segmentation_mask"] = spall_result.masks.xy[j].tolist()
                    
                    output_data["spall_detections"].append(detection_info)

            if (output_data["crack_detections"] or output_data["spall_detections"]):
                json_path = os.path.join(json_folder, f"{image_filename.split('.')[0]}.json")
                try:
                    with open(json_path, 'w') as f:
                        json.dump(output_data, f, indent=4, default=convert_types)
                except Exception as e:
                    print(f"\nError writing JSON for {image_filename}: {e}")
                print(f"{image_filename} has detections and the json file was successfully created")
            else:
                print(f"{image_filename} had no detections")

if __name__ == "__main__":
    process_video_to_frames(video_path, image_folder, desired_fps)
    process_images_to_json(crack_model, spalling_model, image_folder, json_folder, batch_size)