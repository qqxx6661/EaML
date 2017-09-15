#!/usr/bin/env python3
# coding=utf-8
import time
import cv2
from multiprocessing import Process
import csv


class DataCollect(object):
    def __init__(self, cam_id):
        self.cam_id = cam_id

    def data_collect(self):
        # 全局变量
        timecount_start = time.time()
        entropy_last = 0  # 获取参数二：前一帧抖动数值
        point_x, point_y = 0, 0  # 获取参数三：初始化运动点

        with open('HDAdata/' + self.cam_id + '_detection.txt') as file:

            self.row = []

            with open('HDAdata/' + self.cam_id + '_detection_collect.csv', 'w', newline='') as file_new:  # newline不多空行

                f = csv.writer(file_new)

                for line in file:
                    tokens = line.strip().split(',')
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
                    point_x, point_y, smooth, speed_x_last, speed_y_last = self._cal_speed_location(cur_frame,
                                                                                                    point_x,
                                                                                                    point_y,
                                                                                                    smooth,
                                                                                                    speed_x_last,
                                                                                                    speed_y_last)

                    # 写入一行
                    print(self.row)
                    f.writerow(self.row)
                    self.row = []

                    cv2.imshow(str(self.cam_id), cur_frame)

        # 计算总用时，释放内存
        timecount_end = time.time()
        print(self.cam_id, " time:", timecount_end - timecount_start)


def start_collect(cam_id):
    DataCollect(cam_id).data_collect()


if __name__ == "__main__":
    global_start = time.time()
    cam_list = [[50, 0], [53, 1363], [54, ]]
    for i, name in enumerate(cam_list):
        p = Process(target=start_collect, args=(i, name))
        p.start()

    cv2.destroyAllWindows()
    global_end = time.time()
    print("global time:", global_end - global_start)
