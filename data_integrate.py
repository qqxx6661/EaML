#!/usr/bin/env python3
# coding=utf-8
import cv2
import numpy as np
import csv
import time


def output(output_list):

    if len(output_list) == 2:
        if output_list[0] == '0' and output_list[1] == '0':
            return '0'
        if output_list[0] == '1' and output_list[1] == '0':
            return '1'
        if output_list[0] == '0' and output_list[1] == '1':
            return '2'
        if output_list[0] == '1' and output_list[1] == '1':
            return '3'

    if len(output_list) == 4:
        if output_list[0] == '0' and output_list[1] == '0' and output_list[2] == '0' and output_list[3] == '0':
            return '0'
        if output_list[0] == '1' and output_list[1] == '0' and output_list[2] == '0' and output_list[3] == '0':
            return '1'
        if output_list[0] == '0' and output_list[1] == '1' and output_list[2] == '0' and output_list[3] == '0':
            return '2'
        if output_list[0] == '1' and output_list[1] == '1' and output_list[2] == '0' and output_list[3] == '0':
            return '3'
        if output_list[0] == '0' and output_list[1] == '0' and output_list[2] == '1' and output_list[3] == '0':
            return '4'
        if output_list[0] == '0' and output_list[1] == '1' and output_list[2] == '1' and output_list[3] == '0':
            return '5'
        if output_list[0] == '0' and output_list[1] == '0' and output_list[2] == '0' and output_list[3] == '1':
            return '6'
        if output_list[0] == '0' and output_list[1] == '0' and output_list[2] == '1' and output_list[3] == '1':
            return '7'
        # 有特殊帧四个全开，人工识别应该去掉
        if output_list[0] == '1' and output_list[1] == '1' and output_list[2] == '1' and output_list[3] == '1':
            return '8'

def data_integrate(list_file):

    input_each = []
    inputs = []
    outputs = []
    outputs_upper = []

    if len(list_file) == 2:
        for cam_id, file_src in enumerate(list_file):
            if cam_id == 0:  # 双摄像头0号，只需提取右边
                print("读取文件：", file_src)
                with open(file_src) as file:
                    for line in file:
                        tokens = line.strip().split(',')
                        input_each.append(tokens[2])
                        input_each.append(tokens[4])
                        input_each.append(tokens[5])
                        input_each.append(int(tokens[3]) * 100)  # 若为1则填100
                        input_each.append(tokens[7])
                        inputs.append(input_each)
                        input_each = []
                        outputs.append([tokens[1]])  # 先创建为数组
                        outputs_upper.append([tokens[3]])  # 先创建为数组
            if cam_id == 1:
                print("读取文件：", file_src)
                with open(file_src) as file:
                    for line_number, line in enumerate(file):
                        tokens = line.strip().split(',')
                        input_each.append(tokens[2])
                        input_each.append(tokens[4])
                        input_each.append(tokens[5])
                        input_each.append(tokens[6])
                        input_each.append(int(tokens[3]) * 100)  # 若为1则填100
                        inputs[line_number].extend(input_each)
                        input_each = []
                        outputs[line_number].append(tokens[1])
                        outputs_upper[line_number].append(tokens[3])

    if len(list_file) == 4:
        for cam_id, file_src in enumerate(list_file):
            if cam_id == 0:
                print("读取文件：", file_src)
                with open(file_src) as file:
                    for line in file:
                        tokens = line.strip().split(',')
                        input_each.append(tokens[2])
                        input_each.append(tokens[4])
                        input_each.append(tokens[5])
                        input_each.append(int(tokens[3]) * 100)  # 若为1则填100
                        input_each.append(tokens[7])
                        input_each.append(0)
                        input_each.append(0)
                        inputs.append(input_each)
                        input_each = []
                        outputs.append([tokens[1]])  # 先创建为数组
                        outputs_upper.append([tokens[3]])  # 先创建为数组
            if cam_id == 1:
                print("读取文件：", file_src)
                with open(file_src) as file:
                    for line_number, line in enumerate(file):
                        tokens = line.strip().split(',')
                        input_each.append(tokens[2])
                        input_each.append(tokens[4])
                        input_each.append(tokens[5])
                        input_each.append(tokens[6])
                        input_each.append(int(tokens[3]) * 100)  # 若为1则填100
                        input_each.append(tokens[7])
                        input_each.append(0)
                        inputs[line_number].extend(input_each)
                        input_each = []
                        outputs[line_number].append(tokens[1])
                        outputs_upper[line_number].append(tokens[3])
            if cam_id == 2:
                print("读取文件：", file_src)
                with open(file_src) as file:
                    for line_number, line in enumerate(file):
                        tokens = line.strip().split(',')
                        input_each.append(tokens[2])
                        input_each.append(tokens[4])
                        input_each.append(tokens[5])
                        input_each.append(0)
                        input_each.append(tokens[6])
                        input_each.append(int(tokens[3]) * 100)  # 若为1则填100
                        input_each.append(tokens[7])
                        inputs[line_number].extend(input_each)
                        input_each = []
                        outputs[line_number].append(tokens[1])
                        outputs_upper[line_number].append(tokens[3])
            if cam_id == 3:
                print("读取文件：", file_src)
                with open(file_src) as file:
                    for line_number, line in enumerate(file):
                        tokens = line.strip().split(',')
                        input_each.append(tokens[2])
                        input_each.append(tokens[4])
                        input_each.append(tokens[5])
                        input_each.append(0)
                        input_each.append(0)
                        input_each.append(tokens[6])
                        input_each.append(int(tokens[3]) * 100)  # 若为1则填100
                        inputs[line_number].extend(input_each)
                        input_each = []
                        outputs[line_number].append(tokens[1])
                        outputs_upper[line_number].append(tokens[3])
    # print(inputs)
    # print(outputs)
    # print(outputs_upper)

    row = []
    with open('data/train_' + str(len(list_file)) + 'cam.csv', 'a', newline='') as f:  # newline不多空行, a是追加模式
        f_csv = csv.writer(f)
        for i in range(len(outputs)):
            row.append(output(outputs[i]))
            row.append(output(outputs_upper[i]))
            for j in range(len(inputs[0])):
                row.append(inputs[i][j])
            f_csv.writerow(row)
            row = []


if __name__ == "__main__":

    global_start = time.time()
    list_file_name = ["data/2cam_scene1/data_2017-08-07 18-03-28_0.csv",
                      "data/2cam_scene1/data_2017-08-07 18-03-28_1.csv"]
    data_integrate(list_file_name)
    global_end = time.time()
    print("global time:", global_end - global_start)