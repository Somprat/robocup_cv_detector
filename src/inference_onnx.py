import json
from pathlib import Path

import onnxruntime as ort
import cv2
import numpy as np

threshold = 0.25

# Model IDs from sim_data_det_0126.onnx metadata; dataset IDs from
# datasets/hslvision/README.md. None means there is no matching dataset class.
MODEL_TO_HSLVISION = {
    0: 1,     # Ball -> Ball
    1: 2,     # Goalpost -> Goalpost
    2: None,  # Person -> no matching class
    3: 4,     # L_cross -> L-Intersections
    4: 6,     # T_cross -> T-Intersections
    5: 7,     # X_cross -> X-Intersections
    6: 5,     # Penalty_point -> Penalty Marks
    7: None,     # Opponent -> K1 (humanoid robot) might be the case but we are still not sure if opponent means k1
    8: None,  # BR_marker -> no matching class
}

onnx_model_path = "robocup_demo/src/vision/model/sim_data_det_0126.onnx"

ort_session = ort.InferenceSession(onnx_model_path)

img_path = "datasets/hslvision/images/test/4.jpg"
# The annotations file is in the dataset root, above images/test.
annotations_path = Path(img_path).parents[2] / "test_annotations.json"
with annotations_path.open(encoding="utf-8") as annotations_file:
    annotations = json.load(annotations_file)

image_relative_path = Path(img_path).relative_to(annotations_path.parent).as_posix()
image_annotation = next(
    (entry for entry in annotations if entry["local_image"] == image_relative_path),
    None,
)

image = cv2.imread(img_path)

image = cv2.resize(image, (320, 320))
# reshape image to (320, 320, 3)
image = image.transpose(2, 0, 1)[None, ...].astype(np.float32)/255

onnx_output = ort_session.run(None, {"images": image})
onnx_flatten = onnx_output[0][0].transpose(1, 0)
# onnx_flatten has a shape of (2100, 13) = 2100 bound box in one image with 13 values. 
# First4 is x, y, w, h. and the 9 is the confidenc of each class.

max_con = [max(prediction[4:]) for prediction in onnx_flatten]
meta = ort_session.get_modelmeta().custom_metadata_map

exceed_index = [i for i, num in enumerate(max_con) if num >= threshold]

# Pair each model class ID with its bounding box.
bbox = [
    (int(np.argmax(onnx_flatten[i][4:])), onnx_flatten[i][:4])
    for i in exceed_index
]

def output_to_dataset(xywh, scale_x, scale_y):
    left = xywh[0] - xywh[2]/2
    top = xywh[1] - xywh[3]/2
    # in image coordinates, the y value increases downward.
    width = xywh[2]
    height = xywh[3]

    minx_scaled = left/320*scale_x
    width_scaled = width/320*scale_x

    miny_scaled = top/320*scale_y
    height_scaled = height/320*scale_y
    

    return [minx_scaled, miny_scaled, width_scaled, height_scaled]



bbox = [
    (class_id, output_to_dataset(box, 1280, 720))
    for class_id, box in bbox
]
# end goal: design an evaluation pipeline. Comparing this to the threshold
# try printing out bbox from dataset

def nms_filter(bboxes, scores, iou_threshold=0.5):
    """Filter (class_id, [left, top, width, height]) boxes using their scores."""
    if len(bboxes) != len(scores):
        raise ValueError("Each bounding box must have one confidence score")

    keep = []
    for class_id in sorted({class_id for class_id, _ in bboxes}):
        indices = [i for i, (label, _) in enumerate(bboxes) if label == class_id]
    
        # got a list of index position of every bound box for one class. 
        # Classid would change in each iteration of for loop
        # from a bbox that contains every classes not just from this class.
        
        # input the list of bound box and its score
        selected = cv2.dnn.NMSBoxes(
            [list(map(float, bboxes[i][1])) for i in indices],
            [float(scores[i]) for i in indices],
            score_threshold=0.0,
            nms_threshold=iou_threshold,
        )
        # 

     
        # only 2/5 boxes survive filtering
        keep.extend(indices[int(i)] for i in np.asarray(selected).flatten())
        # append the 4th and the seond bounding box's id
    # return the bbox at position i. so one bbox = (class_id, bound_box_position)
    result = [bboxes[i] for i in sorted(keep, key=lambda i: scores[i], reverse=True)]
    return result


# Confidence scores in the same order as bbox, before NMS.
scores = [float(max_con[i]) for i in exceed_index]

bbox_filtered = nms_filter(bbox, scores)
bbox_maped = [(MODEL_TO_HSLVISION[pre_class], pre_bbox) for (pre_class, pre_bbox) in bbox_filtered]

image_selected = [annotation for annotation in annotations if annotation['local_image'] == 'images/test/4.jpg'][0]

# a list of pair of class and bbox fro annotations
corresponding_annot = []
for i in range(len(image_selected['objects']['category'])):
    corresponding_annot.append((image_selected['objects']['category'][i], image_selected['objects']['bbox'][i]))

# Next
# 1. IOU
# 2. TP NP ...
# Done!

# do tp, fp,.. by matching iou

# Match per image and class: process predictions from highest confidence to lowest. Match each prediction to the unmatched annotation with the highest intersection-over-union (IoU), accepting it if IoU ≥ 0.5. Each annotation can match only once.
# Count results:- Matched prediction → true positive (TP)
# - Unmatched prediction → false positive (FP)
# - Unmatched annotation → false negative (FN)

# Report per-class precision and recall across the test set:- Precision = TP / (TP + FP)
# - Recall = TP / (TP + FN)