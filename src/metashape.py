import Metashape
import os
import json
from shapely.geometry import Polygon


json_folder = r"C:\Programming\crack_detection\videos\vid1\vid1_json"
layer_name = "Projected Damage"
confidence_threshold = .2
iou_threshold = .4

def check_for_duplicate_masks(new_coords, polygons, iou_threshold):
    new_polygon = Polygon(new_coords)
    if not polygons:
        polygons.append(new_polygon)
        return False

    for polygon in polygons:
        intersection = new_polygon.intersection(polygon)
        union = new_polygon.union(polygon)
        if union.area == 0:
            continue
        iou_score = intersection.area / union.area
        if (iou_score > iou_threshold):
            return True
        
    polygons.insert(0, new_polygon)
    return False

def project_to_model(json_folder, layer_name, confidence_threshold=.2, iou_threshold=.4, doc=None):
    if not os.path.isdir(json_folder):
        print(f"Folder {json_folder} does not exist")
        return

    if doc is None:
        doc = Metashape.app.document
    chunk = doc.chunk
    if not chunk:
        print("No chunk found in the document")
        return

    if not chunk.shapes:
        chunk.shapes = Metashape.Shapes()

    surface = chunk.model

    #chunk.crs = Metashape.app.getCoordinateSystem("Select Coordinate System", chunk.crs)
    chunk.shapes.crs = chunk.crs
    
    matrix_t = chunk.transform.matrix
    
    # Check if group already exists
    existing_group = None
    for group in chunk.shapes.groups:
        if group.label == layer_name:
            existing_group = group
            break
    
    if existing_group:
        group = existing_group
        print(f"Using existing group: {layer_name}")
    else:
        group = chunk.shapes.addGroup()
        group.label = layer_name
        group.color = (255, 0, 0)
        print(f"Created new group: {layer_name}")

    shapes_added = 0
    polygons = [] #For checking duplicate detections
    
    for camera in chunk.cameras:
        if not camera.transform:
            print(f"Skipping {camera.label} - no transform")
            continue
            
        json_file = os.path.join(json_folder, f"{camera.label}.json")

        #If there are no detections, JSON file is not created and is skipped
        if not os.path.exists(json_file):
            continue

        print(f"JSON file found: {json_file}")
        try:
            with open(json_file, 'r') as f:
                detections = json.load(f)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue

        for detection_type in ['crack_detections', 'spall_detections']:
            for detection in detections.get(detection_type, []):
                if 'segmentation_mask' in detection:
                    try:
                        # # Convert 2D image coordinates to 3D world coordinates
                        # # Assuming points are in [x, y] format in image coordinates
                        
                        points_2d = detection['segmentation_mask']
                        confidence = detection['confidence']
                        if (confidence < confidence_threshold): continue
                        projected_points = []

                        for p in points_2d:
                            point = surface.pickPoint(camera.center, camera.unproject(Metashape.Vector(p)))
                            if point:
                                projected_points.append(chunk.crs.project(matrix_t.mulp(point)))

                        if len(projected_points) > 2:
                            if (check_for_duplicate_masks([tuple(vec) for vec in projected_points], polygons, iou_threshold)):
                                continue
                            # if (len(polygons) > 50):
                            #     polygons.pop()
                            shape = chunk.shapes.addShape()
                            shape.label = f"{detection_type.split('_')[0]} - {confidence:.2f}"
                            shape.group = group
                            shape.boundary_type = Metashape.Shape.BoundaryType.OuterBoundary
                            shape.geometry = Metashape.Geometry.Polygon(projected_points)
                            shapes_added += 1
                        
                    except Exception as e:
                        print(f"Error creating shape for {camera.label}: {e}")

    print(f"Added {shapes_added} shapes to group '{layer_name}'")
    
    if shapes_added > 0:
        doc.save()
        print("Document saved")
    else:
        print("No shapes were added")

    doc.save()

if __name__ == "__main__":
    project_to_model(json_folder, layer_name, confidence_threshold, iou_threshold)