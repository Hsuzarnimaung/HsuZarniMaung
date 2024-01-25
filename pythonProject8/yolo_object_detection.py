import cv2
import numpy as np

global input
import tkinter as tk
from PIL import ImageTk, Image


# Load Yolo
def yolo(weight, detection_score, entry_path):
    net = cv2.dnn.readNet("weights/dataset2_80-20/" + weight, "weights/dataset2_80-20/yolov4-custom.cfg")
    classes = []
    with open("weights/dataset2_80-20/obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if input == "image":
        # Loading image
        img = cv2.imread(entry_path)
        img = cv2.resize(img, None, fx=0.8, fy=0.8)
        height, width, channels = img.shape
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)
        print(outs)
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[int(detection_score):]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]] + " " + str(round(confidences[i], 2)))
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, label, (x, y - 10), font, 2, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif input == "video":
        # Open the video file
        cap = cv2.VideoCapture(entry_path)
        while True:
            _, img = cap.read()
            height, width, channels = img.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)
            print(outs)
            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[int(detection_score):]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            print(indexes)
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]] + " " + str(round(confidences[i], 2)))
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, label, (x, y - 10), font, 2, (0, 0, 255), 2)
            # Calculate the position of the image window
            screen_width = 400  # Replace with the desired screen width
            screen_height = 600  # Replace with the desired screen height
            window_x = int((screen_width - width) / 2)
            window_y = int((screen_height - height) / 2)

            # Create a window and display the image at the center
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Image", window_x, window_y)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif input == "webcam":
        # Open the webcam
        cap = cv2.VideoCapture(0)
        while True:
            _, img = cap.read()
            height, width, channels = img.shape
            # width = 512
            # height = 512

            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)
            print(outs)

            # Showing information on the screen
            class_ids = []
            confidences = []

            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            print(indexes)
            if indexes == 0: print("weapon detected in frame")
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]]) + " " + str(round(confidences[i], 2))

                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y - 10), font, 0.6, (255, 255, 255), 2)

            frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imshow("Image", img)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Invalid input type. Please choose 'image', 'video', or 'webcam'.")
