#!/usr/bin/env python3
# coding=utf-8
import cv2
import datetime
import numpy as np
import csv
import time
from multiprocessing import Process


class DataCollect(object):
    def __init__(self, cam_id, video_name):
        self.cam_id = cam_id
        self.video_name = video_name
        self.row = []

    def _judge_move(self, cur_frame_inner, pre_frame_inner):
        # gray_img = cv2.cvtColor(cur_frame_inner, cv2.COLOR_BGR2GRAY)
        gray_img = cur_frame_inner
        gray_img = cv2.resize(gray_img, (500, 500))  # 此条不知是否影响判断
        gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)
        if pre_frame_inner is None:
            pre_frame_inner = gray_img
            return pre_frame_inner
        else:
            img_delta = cv2.absdiff(pre_frame_inner, gray_img)
            thresh = cv2.threshold(img_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            # image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < 500:  # 设置敏感度
                    continue
                else:
                    # print(cv2.contourArea(c))
                    # print("画面中有运动物体")
                    self.row.pop()
                    self.row.append('1')
                    break
            pre_frame_inner = gray_img
            return pre_frame_inner

    def _entropy(self, band):  # 计算画面熵
        hist, _ = np.histogram(band, bins=range(0, 256))
        hist = hist[hist > 0]
        return -np.log2(hist / hist.sum()).sum()

    def _process_rgb_delta(self, cur_frame_inner, entropy_last_inner):  # 计算熵抖动
        # b, g, r = cv2.split(cur_frame_inner)
        # rgb_average = (self._entropy(r) + self._entropy(g) + self._entropy(b)) / 3
        gray_average = self._entropy(cur_frame_inner)
        if entropy_last_inner == 0:
            self.row.append(0)
            return gray_average
        jitter = abs(gray_average - entropy_last_inner)
        jitter = int(jitter)
        # print("画面抖动数值:", jitter)
        self.row.append(jitter)
        return gray_average

    def _cal_speed_location(self, cur_frame_inner, point_x_inner, point_y_inner):

        # cur_frame_inner = cv2.cvtColor(cur_frame_inner, cv2.COLOR_BGR2GRAY)
        bodycascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
        bodys = bodycascade.detectMultiScale(
            cur_frame_inner,
            scaleFactor=1.05,  # 越小越慢，越可能检测到
            minNeighbors=2,  # 越小越慢，越可能检测到
            minSize=(95, 80),
            maxSize=(150, 180),
            # minSize=(30, 30)
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(bodys) == 0:  # 没有人脸，速度和摄像头都为0
            self.row.append(0)  # 一个自身判断无运动
            self.row.append(0)  # 两个速度
            self.row.append(0)
            self.row.append(0)  # 两个方向
            self.row.append(0)

        else:

            self.row.append(1)  # 自身判断有运动
            # 只输入第一张人脸数据
            print('Now face:', bodys)
            x, y, w, h = bodys[0][0], bodys[0][1], bodys[0][2], bodys[0][3]
            p1 = (x, y)
            p2 = (x + w, y + h)
            cv2.rectangle(cur_frame_inner, p1, p2, (0, 255, 0), 2)

            if point_x_inner == 0 and point_y_inner == 0:  # 刚标记后第一帧
                # 两个速度为0
                self.row.append(0)
                self.row.append(0)
            else:
                v_updown = point_y_inner - p1[1]
                v_leftright = p1[0] - point_x_inner
                # print("横轴速度为：", v_leftright)
                # print("纵轴速度为：", v_updown)
                self.row.append(v_leftright)
                self.row.append(v_updown)

            point_x_inner = p1[0]
            point_y_inner = p1[1]

            if p1[0] <= 50:
                if p1[0] < 0:
                    self.row.append(50)
                else:
                    self.row.append(50 - p1[0])
                print("左边该开了", 50 - p1[0])
            else:
                self.row.append(0)

            if p2[0] >= 590:
                if p2[0] > 640:
                    self.row.append(50)
                else:
                    self.row.append(p2[0] - 590)
                    print("右边该开了", p2[0] - 590)
            else:
                self.row.append(0)

        return point_x_inner, point_y_inner

    def data_collect(self):
        # 全局变量
        timecount_start = time.time()
        time_stamp = 1  # 时间标记
        pre_frame = None  # 获取参数一：前一帧图像（灰度），判断是否有运动物体
        entropy_last = 0  # 获取参数二：前一帧抖动数值
        point_x, point_y = 0, 0  # 获取参数三：初始化运动点

        camera = cv2.VideoCapture(self.video_name)

        self.row = []
        file_name = str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + '_' + str(self.cam_id))
        with open('data/data_' + file_name + '.csv', 'w', newline='') as file:  # newline不多空行
            f = csv.writer(file)

            # 循环获取参数
            while True:
                res, cur_frame = camera.read()
                if res is not True:
                    break
                cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)  # 转变至Gray

                if cv2.waitKey(1) & 0xFF == 27:
                    break

                # 参数0：时间(暂时加入时间帧)
                # time_now = str(datetime.datetime.now().strftime("%H%M%S%f"))
                # row.append(time_now[:-4])  # 毫秒只取两位
                self.row.append(time_stamp)
                time_stamp += 1
                print('------', time_stamp, '-------')

                # 获取参数一：开/关
                self.row.append('0')  # 判断有无运动，遇到有物体运动再改为1
                pre_frame = self._judge_move(cur_frame, pre_frame)

                # 获取参数二：图像抖动
                entropy_last = self._process_rgb_delta(cur_frame, entropy_last)

                # 获取参数三：速度和对应摄像头开关
                point_x, point_y = self._cal_speed_location(cur_frame, point_x, point_y)

                # 写入一行
                print(type(self.row), self.row)
                f.writerow(self.row)
                self.row = []

                cv2.imshow(str(self.cam_id), cur_frame)

        # 计算总用时，释放内存
        timecount_end = time.time()
        print(self.cam_id, " time:", timecount_end - timecount_start)
        camera.release()


def start_collect(cam_id, video_name):
    DataCollect(cam_id, video_name).data_collect()

if __name__ == "__main__":

    global_start = time.time()
    list_video_name = ["video/2cam_scene1/2017-08-07 17-54-50_0.avi",
                       "video/2cam_scene1/2017-08-07 17-54-50_1.avi"]

    for i, name in enumerate(list_video_name):
        p = Process(target=start_collect, args=(i, name))
        p.start()

    cv2.destroyAllWindows()
    global_end = time.time()
    print("global time:", global_end - global_start)
