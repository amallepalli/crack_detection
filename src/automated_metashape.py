import os
import Metashape
from ultralytics import YOLO
from metashape import project_to_model
from save_segmentation_results import process_video_to_frames, process_images_to_json

video_path = r"C:\Programming\crack_detection\videos\vid1\vid1.MP4"
desired_fps = 2

crack_model = YOLO("C:\Programming\crack_detection\models\crack_segmentation_model_02.pt")      
spalling_model = YOLO("C:\Programming\crack_detection\models\spalling_segmentation_model_01.pt")

project_file_name = r"C:\Programming\crack_detection\videos\vid1\vid1.psx"
image_folder = r"C:\Programming\crack_detection\videos\vid1\vid1_frames"
json_folder = r"C:\Programming\crack_detection\videos\vid1\vid1_json"
batch_size = 10
layer_name = "Projected Damage"
confidence_threshold = .2
iou_threshold = .4

def automate_one_video(video_path, desired_fps, crack_model, spalling_model, batch_size, layer_name, 
                       project_file_path=None, image_folder=None, json_folder=None, confidence_threshold=.2, iou_theshold=.7):
    if project_file_path is None:
        project_file_path = os.path.splitext(video_path) + ".psx"
    
    if image_folder is None:
        image_folder = os.path.splitext(video_path) + "_frames"

    if json_folder is None:
        json_folder = os.path.splitext(video_path) + "_json"
    
    process_video_to_frames(video_path, image_folder, desired_fps)
    process_images_to_json(crack_model, spalling_model, image_folder, json_folder, batch_size)
    
    doc = Metashape.Document()
    doc.save(project_file_name)
    chunk = doc.addChunk()

    # Saves after every step so if there is a failure, program can be recovered
    # where it left off by commenting out already completed steps

    chunk.addPhotos([os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))])
    doc.save()
    chunk.matchPhotos()
    doc.save()
    chunk.alignCameras()
    doc.save()
    chunk.buildDepthMaps()
    doc.save()
    chunk.buildModel()
    doc.save()
    chunk.buildUV()
    doc.save()
    chunk.buildTexture()
    doc.save() 

    project_to_model(json_folder, layer_name, confidence_threshold, iou_theshold, doc=doc)


def automate_video_folder(video_folder, desired_fps, crack_model, spalling_model, batch_size, layer_name, 
                       project_file_path=None, image_folder=None, json_folder=None, confidence_threshold=.2, iou_theshold=.7):
    if not os.path.isdir(video_folder):
        print(f"Folder {video_folder} does not exist")
        return

    for video_path in os.listdir(video_folder):
        automate_one_video(os.path.join(video_folder, video_path), desired_fps, crack_model, spalling_model, 
                           batch_size, layer_name, confidence_threshold=confidence_threshold, iou_theshold=iou_threshold)

if __name__ == "__main__":
    #Metashape.License().activate("T7BX9-S9XJC-SBBTV-3N6RO-2TH33")
    automate_one_video(video_path, desired_fps, crack_model, spalling_model, batch_size, layer_name, confidence_threshold=confidence_threshold, iou_theshold=iou_threshold)