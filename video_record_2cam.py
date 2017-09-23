#!/usr/bin/env python3
# coding=utf-8
import datetime
import cv2
import time
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
# 定义解码器并创建VideoWrite对象
# linux: XVID、X264; windows:DIVX
# 20.0指定一秒钟的帧数
fourcc = cv2.VideoWriter_fourcc(*'XVID')
file_name = str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
out0 = cv2.VideoWriter('video/' + file_name + '_0.avi', fourcc, 30.0, (640, 480))
out1 = cv2.VideoWriter('video/' + file_name + '_4.avi', fourcc, 30.0, (640, 480))
record_status = 1
record_total = 1800
time.sleep(30)
while True:
    # 读取一帧
    status0, frame0 = cap0.read()
    status1, frame1 = cap1.read()


    if status0 is not True:
        print('0 failed')
        break
    if status1 is not True:
        print('1 failed')
        break
    if cv2.waitKey(10) & 0xFF == 27:
        break

    if record_status == 1:
        # frame0 = cv2.flip(frame0, 0)
        # frame1 = cv2.flip(frame1, 0) # 翻转图像
        out0.write(frame0)
        out1.write(frame1)

        print(record_total)
        if record_total == 0:
            break
        record_total -= 1

    # 显示帧
    cv2.imshow('frame0', frame0)
    cv2.imshow('frame1', frame1)

    # if cv2.waitKey(5) & 0xFF == ord('r'):
    #     record_status = 1
    #     print("Start recording")
    #     continue
    # if cv2.waitKey(5) & 0xFF == ord('t'):
    #     record_status = 0
    #     print("Stop recording")
    #     continue


# 释放VideoCapture对象
cap0.release()
cap1.release()
out0.release()
out1.release()
cv2.destroyAllWindows()

