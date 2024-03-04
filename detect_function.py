import cv2
import numpy as np


def perform_object_detection(video_path, output_path):
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    person = classes.index("person")

    video = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    while video.isOpened():
        ret, img = video.read()
        if not ret:
            break
        img, boxes = detect_objects(img, net, classes, colors, person)
        img = draw_boxes(img, boxes)
        cv2.imshow('object_detection', img)

        key = cv2.waitKey(1)
        if key == 32:  # Space
            key = cv2.waitKey(0)
        if key == 27:  # ESC
            break

    video.release()
    video_writer.release()
    cv2.destroyAllWindows()


def detect_objects(img, net, classes, colors, target_class_index):
    img = cv2.resize(img, None, fx=0.7, fy=0.7)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == target_class_index:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return img, boxes


def draw_boxes(img, boxes):
    font = cv2.FONT_HERSHEY_PLAIN
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        label = "Person"
        cv2.putText(img, label, (x, y + 30), font, 2, (255, 0, 0), 1)
    return img



perform_object_detection('test_video.mp4', 'output_detected_video.mp4')