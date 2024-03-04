import cv2
import numpy as np
import overlap as ol
# from PIL import Image, ImageDraw

#ROI

#tracker init
trackers = [cv2.TrackerBoosting_create,
            cv2.TrackerMIL_create,
            cv2.TrackerKCF_create,
            cv2.TrackerTLD_create,
            cv2.TrackerMedianFlow_create,
            cv2.TrackerGOTURN_create,
            cv2.TrackerCSRT_create,
            cv2.TrackerMOSSE_create]

trackerIdx = 2
tracker = None

# Yolo 로드
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

person = classes.index("person")

src = 'test_video.mp4'

cap = cv2.VideoCapture(src) #영상불러옴

fps = cap.get(cv2.CAP_PROP_FPS)
# print(fps)



delay = int(1000 / fps)
win_name = 'Tracking APIs'

if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret == False:
            print("End of Video")
            break
        frame = cv2.resize(img, dsize=None, fx=0.7, fy=0.7)
        height, width, channels = frame.shape
        img_draw = frame.copy()


        # rect_top_left = (480, 210)
        # rect_bottom_right = (560, 290)
        # cv2.putText(img_draw, "safe zone.", (472, 205), \
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        # # crop_roi 생성
        # crop_roi = img_draw[rect_top_left[1]:rect_bottom_right[1], rect_top_left[0]:rect_bottom_right[0]]
        # crop_roi[:, :, 1] = 200

        # crop_roi[1, 1, 1] = 0

        #tracker가 할당 됐을 경우에만
        if tracker is None:
            rect_top_left = (400, 210)
            rect_bottom_right = (470, 280)
            crop_roi = img_draw[rect_top_left[1]:rect_bottom_right[1], rect_top_left[0]:rect_bottom_right[0]]
            crop_roi[:, :, 1] = 200
            cv2.putText(img_draw, "safe zone.", (400, 205), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.rectangle(img_draw, (480, 210), (560, 290), (255, 255, 255), 2)
            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers) # 감지 결과. 개체에 대한 모든 정보와 위치 제공
            # 정보를 화면에 표시
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and class_id == person:
                        # Object detected
                        print(confidence)
                        center_x = int(detection[0] * width)
                        print("center_x",center_x)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # 좌표
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # 노이즈 제거

            # 화면 출력
            font = cv2.FONT_HERSHEY_PLAIN


            for i in range(len(boxes)):

                if i in indexes:
                    x, y, w, h = boxes[i]

                    label = str(classes[class_ids[i]])
                    cv2.rectangle(img_draw, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    cv2.putText(img_draw, label, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

                    if ol.overlap(boxes[i]) == True:
                        print(boxes[i])
                        bbox = (x, y, w, h)
                        #tracker 할당 부분
                        if boxes[2] and boxes[3]:
                             tracker = trackers[trackerIdx]()
                             isInit = tracker.init(img_draw, bbox)

        else:


            print("got it!")
            # crop_roi = img_draw[rect_top_left[1]:rect_bottom_right[1], rect_top_left[0]:rect_bottom_right[0]]
            # crop_roi = None

            ok, bbox = tracker.update(frame)
            print(ok, bbox)
            (x, y, w, h) = bbox

            if ok:  # 추적 성공

                cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)),
                              (0, 255, 0), 2, 1)
                cv2.putText(img_draw, "safe zone.", (int(x), int(y)), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)



        trackerName = tracker.__class__.__name__

        # 원래위치
        # cv2.putText(img_draw, "safe zone.", (472, 205), \
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)



        cv2.imshow(win_name, img_draw)

        key = cv2.waitKey(100) & 0xff



else:
    print("can't open video")


cap.release()
cv2.destroyAllWindows()



