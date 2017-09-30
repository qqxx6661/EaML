#!/usr/bin/env python3
# coding=utf-8
# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import csv


def reID_extractor(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    print(type(hist))
    return hist


if __name__ == '__main__':

    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    filename = 'video_BSU/2017-09-23 15-47-32_5.avi'
    vs = FileVideoStream(filename).start()
    time.sleep(2.0)
    fps = FPS().start()

    # create CSV
    row = []
    timestamp = 0
    cam_id = filename[-5]
    with open('BSU_data/' + cam_id + '.csv', 'w', newline='') as file:  # newline不多空行
        f = csv.writer(file)

        # loop over the frames from the video stream
        for i in range(1800):  # while 会卡住

            person_count = 0
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            frame = vs.read()

            frame = imutils.resize(frame, width=400)

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            row.append(int(cam_id))
            row.append(timestamp)


            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > 0.7:
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the prediction on the frame
                    if CLASSES[idx] == 'person':

                        # # cut the ROI image
                        person_image = frame[startY:endY, startX:endX]

                        # 剪切图片保存（尝试）
                        startX_new = int(startX + 0.25 * (endX - startX))
                        endX_new = int(endX - 0.25 * (endX - startX))
                        print(startX, startY, endX, endY)
                        print(startX_new, startY, endX_new, endY)
                        person_image = frame[startY:endY, startX_new:endX_new]


                        # 保存为nparray
                        reID_feature = reID_extractor(person_image)  # class 'numpy.ndarray'
                        reID_filename = 'BSU_data/' + cam_id + '/' + str(timestamp) + '_' + str(person_count) + '.npy'
                        np.save(reID_filename, reID_feature)

                        # 显示
                        # try:
                        #     cv2.imshow('image', person_image)
                        #     cv2.waitKey(0)
                        # except:
                        #     continue

                        # 保存为图片
                        # cv2.imwrite('reID_image_test/' + cam_id + '_image' + str(timestamp) + '.jpg',
                        #             person_image)

                        row.append([[startX, startY, endX, endY], reID_filename])

                        label = str(timestamp) + "{}: {:.2f}%".format(CLASSES[idx],
                            confidence * 100)
                        # 改为显示裁剪后box
                        cv2.rectangle(frame, (startX_new, startY), (endX_new, endY),
                            COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # update the FPS counter
            fps.update()

            print(row)
            f.writerow(row)
            row = []
            timestamp += 1

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()