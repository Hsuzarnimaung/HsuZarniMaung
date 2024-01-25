import os
import cv2
import numpy as np
from podm.metrics import get_pascal_voc_metrics, MetricPerClass, get_bounding_boxes
from podm.box import Box, intersection_over_union
def calculate_ap(tp, fp, fn):
    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision * recall

def calculate_mAP(total_tp, total_fp, total_fn, num_classes):
    # Replace this with the actual number of classes in your dataset
    ap_sum = 0.0

    for class_id in range(num_classes):
        tp = total_tp[class_id]
        fp = total_fp[class_id]
        fn = total_fn[class_id]

        ap = calculate_ap(tp, fp, fn)
        ap_sum += ap

    mAP = ap_sum / num_classes
    return mAP

def calculate_ciou(boxA, boxB):
    # Extract coordinates of boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate intersection area
    intersection_area = max(0, xB - xA) * max(0, yB - yA )

    # Calculate box areas
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Calculate Union area by subtracting the intersection area
    # and adding the areas of both boxes
    union_area = boxA_area + boxB_area - intersection_area

    # Calculate box center points
    boxA_center_x = (boxA[0] + boxA[2]) / 2
    boxA_center_y = (boxA[1] + boxA[3]) / 2
    boxB_center_x = (boxB[0] + boxB[2]) / 2
    boxB_center_y = (boxB[1] + boxB[3]) / 2

    # Calculate box diagonal distance
    boxA_diagonal = np.sqrt(np.square(boxA[2] - boxA[0]) + np.square(boxA[3] - boxA[1]))
    boxB_diagonal = np.sqrt(np.square(boxB[2] - boxB[0]) + np.square(boxB[3] - boxB[1]))

    # Calculate distance between box centers
    center_distance = np.square(boxA_center_x - boxB_center_x) + np.square(boxA_center_y - boxB_center_y)

    # Calculate complete IoU (CIoU)
    ciou = intersection_area / union_area - center_distance / np.square(max(boxA_diagonal, boxB_diagonal))

    return ciou


start_time_inference = cv2.getTickCount()
# Start the timer before post-processing
start_time_post_processing = cv2.getTickCount()
def evaluate_image(image_path, net, output_layers, classes, colors, detection_threshold=0.5):
    # Loading image
    image = cv2.imread(image_path)

    Width = image.shape[1]
    Height = image.shape[0]
    #print(Width,Height)
    # Collecting detections
    class_ids = []
    confidences = []
    boxes = []
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    dection = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence >= 0.25:
                x = int(detection[0] * Width)
                y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                class_ids.append(class_id)
                #print(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

                #print(dection)
    # Non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.4)

    # Draw bounding boxes and labels
    font = cv2.FONT_HERSHEY_PLAIN
    class_ID=[]
    for i in indices:
        #i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        label = str(classes[class_ids[i]] + " " + str(round(confidences[i], 2)))
        color = colors[class_ids[i]]
        x1 = x - w / 2
        y1 = y - h / 2
        dection.append([x, y, x + w, y + h])
        #print(dection)
        cv2.rectangle(image, (int(x1),int(y1)), (int(x1 + w),int(y1 + h)), (255, 0, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), font, 2, (255, 0, ), 2)
        class_ID.append(class_ids[i])
    return image, dection, class_ID
# Function to calculate Intersection over Union (IoU) between two bounding boxes

# Path to the image dataset folder
dataset_folder = "C:/Users/Yon Mi Mi Hlaing/Desktop/yoon/90-10/test1"

# Output folder for saving detected objects
output_folder_tp = "C:/Users/Yon Mi Mi Hlaing/Desktop/yoon/pythonProject2/90-10 testing/8000_c0.25_0.4_i0/ntp"
os.makedirs(output_folder_tp, exist_ok=True)
output_folder_fp = "C:/Users/Yon Mi Mi Hlaing/Desktop/yoon/pythonProject2/90-10 testing/000_c0.25_0.4_i0/nfp"
os.makedirs(output_folder_fp, exist_ok=True)
output_folder_fn = "C:/Users/Yon Mi Mi Hlaing/Desktop/yoon/pythonProject2/90-10 testing/8000_c0.25_0.4_i0/nfn"
os.makedirs(output_folder_fn, exist_ok=True)

# Load the YOLOv4 model and other required variables
net = cv2.dnn.readNet("C:/Users/Yon Mi Mi Hlaing/Desktop/yoon/pythonProject2/weights/dataset2_90-10/yolov4-custom_8000.weights", "C:/Users/Yon Mi Mi Hlaing/Desktop/yoon/pythonProject2/weights/dataset2_80-20/yolov4-custom.cfg")
classes = []
with open("C:/Users/Yon Mi Mi Hlaing/Desktop/yoon/pythonProject2/weights/dataset2_80-20/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

total_tp = 0
total_tn = 0
total_fp = 0
total_fn = 0
total_TPs= [0]*len(classes)
total_FNs=[0]*len(classes)
total_FPs =[0]*len(classes)
total_TNs=[0]*len(classes)
print(total_TPs)
print(len(classes))
# Initialize counters for each image
tp = tn = fp = fn = 0
# Iterate over images in the dataset folder
for image_file in os.listdir(dataset_folder):

    if image_file.endswith(".jpg") or image_file.endswith(".png") or image_file.endswith(".JPG"):
        image_path = os.path.join(dataset_folder, image_file)
        # Start the timer before performing inference

        image = cv2.imread(image_path)

        Width = image.shape[1]
        Height = image.shape[0]
        # Evaluate image and obtain detections
        output_image, detections, class_ids = evaluate_image(image_path, net, output_layers, classes, colors)

        # Perform counting based on ground truth (assuming annotations exist in a separate folder)
        ground_truth_file = os.path.splitext(image_file)[0] + ".txt"
        ground_truth_path = os.path.join("C:/Users/Yon Mi Mi Hlaing/Desktop/yoon/90-10/test1", ground_truth_file)
        # Compare detections with ground truth annotations

        # Load ground truth annotations
        with open(ground_truth_path, "r") as f:
            lines = f.readlines()

        # Extract ground truth bounding boxes
        ground_truth_boxes = []
        for line in lines:
            values = line.split()
            class_id = int(values[0])
            x, y, w, h = map(float, values[1:])
            ground_truth_boxes.append([class_id, (x*Width), (y*Height),((x*Width)+ (w*Width)), ((y*Height)+(h*Height))])
            #print(ground_truth_boxes)
        matched_ground_truth = [False] * len(ground_truth_boxes)
        #print(len(detections))
        op_image = output_image
        for ground_truth_box in ground_truth_boxes:
            w = ground_truth_box[3] - ground_truth_box[1]
            h = ground_truth_box[4] - ground_truth_box[2]
            x1 = ground_truth_box[1] - w / 2
            y1 = ground_truth_box[2] - h / 2
            cv2.rectangle(op_image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)
            cv2.putText(op_image, classes[ground_truth_box[0]] + "__GT",
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0),
                        2)
        # print(len(detections))
        for detected_box, detected_class_id in zip(detections, class_ids):

            # Calculate IoU (Intersection over Union) for the detected box with each ground truth box
            max_iou = 0
            max_iou_index = -1

            for i, ground_truth_box in enumerate(ground_truth_boxes):
                box1 = [detected_box[0], detected_box[1], detected_box[2], detected_box[3]]
                box2 = [ground_truth_box[1], ground_truth_box[2], ground_truth_box[3],ground_truth_box[4]]
                ciou = calculate_ciou(box1, box2)
                if ciou > max_iou:
                    max_iou = ciou
                    #print(max_iou)
                    max_iou_index = i

            if max_iou >= 0 and detected_class_id == ground_truth_boxes[max_iou_index][0]:
                if not matched_ground_truth[max_iou_index]:
                    tp += 1
                    total_TPs[detected_class_id]+=1
                    print(max_iou)
                    output_path = os.path.join(output_folder_tp, image_file)
                    cv2.imwrite(output_path, output_image)
                    matched_ground_truth[max_iou_index] = True
                else:
                    fp += 1
                    total_FPs[detected_class_id]+=1
                    output_path = os.path.join(output_folder_fp, image_file)
                    cv2.imwrite(output_path, output_image)
            else:
                fp += 1
                total_FPs[detected_class_id] += 1
                output_path = os.path.join(output_folder_fp, image_file)
                cv2.imwrite(output_path, output_image)

            # Calculate remaining unmatched ground truth boxes as false negatives
        FN= sum(not matched for matched in matched_ground_truth)
        if FN>0:
            output_path = os.path.join(output_folder_fn, image_file)
            cv2.imwrite(output_path, output_image)
        fn+=FN
        for matched,gtb in zip(matched_ground_truth,ground_truth_boxes):
            if not matched:
                total_FNs[gtb[0]] += 1


    # Accumulate counts for all images
total_tp += tp
total_fp += fp
total_fn += fn

# Print the total counts
print("True Positives (TP):", total_tp)
print("False Positives (FP):", total_fp)
print("False Negatives (FN):", total_fn)
print(total_TPs)
print(total_FPs)
print(total_FNs)
mAP=calculate_mAP(total_TPs,total_FPs,total_FNs, len(classes))
total_predictions = tp + tn + fp + fn
#mAP = tp / (tp + fp + fn)
accuracy = (tp + tn) / total_predictions
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

end_time_inference = cv2.getTickCount()
elapsed_time_inference = (end_time_inference - start_time_inference) / cv2.getTickFrequency()
# End the timer after post-processing
end_time_post_processing = cv2.getTickCount()
elapsed_time_post_processing = (end_time_post_processing - start_time_post_processing) / cv2.getTickFrequency()

# Print the inference time and post-processing time
#print("Image:", image_file)
print("Inference Time (seconds):", elapsed_time_inference)
print("Post-processing Time (seconds):", elapsed_time_post_processing)

# Print the evaluation metrics
print("Mean Average Precision (mAP):", mAP)
print("Accuracy:", accuracy)
print("F1 Score:", f1_score)
print("Recall:", recall)
print("Precision:", precision)
