#!/usr/bin/env python3
# coding=utf-8
import csv


def relative_position(cood):
    # 输入两点坐标的list，其实就是矩形的左上角和右下角坐标。求该点在画面内的相对位置（左右来看）。视频分辨率为640x480。
    # 举例：该点x坐标是320，那他的相对位置就是0
    value = ((0.5 * (cood[0] + cood[2])) - 320) / 320
    value = int(value * 100)
    return value


person = 'person_0'  # 文件名
cal_speed_delay = 5  # 连续在同一摄像头n帧后再计算速度
cal_speed_delay_flag = 1  # 连续在同一摄像头n帧都有数据则置为1
# 创建所有帧数组
all_data = []
for frame in range(1800):
    all_data.append([frame, ])

# 读取
with open('gallery/' + person +'.csv') as csvFile:
    reader = csv.reader(csvFile)
    for item in reader:
        frame_now = int(item[0])  # 当前处理帧

        # 长度大于1说明这帧之前有了，异常.由于暂时一共就两个人，所以只比较两个值
        if len(all_data[frame_now]) > 1:
            # 这里若报错，说明前面一帧也没有信息，假设没有出现这种情况
            last_position = int(all_data[frame_now - 1][2])
            if abs(all_data[frame_now][2] - last_position) > abs(relative_position(eval(item[2])) - last_position):
                # 删除，未完成
                all_data[frame_now].pop(1)
                all_data[frame_now].pop(2)
                all_data[frame_now].pop(3)
            else:
                # 不需要存储了，使用上次数据
                continue

        all_data[frame_now].append(item[1])  # 加入camid
        all_data[frame_now].append(relative_position(eval(item[2])))  # 加入相对位置
        # 加入速度：连续N帧有，才加入速度
        for i in range(cal_speed_delay):
            # print(len(all_data[frame_now - (i+1)]))
            if len(all_data[frame_now - (i+1)]) == 1:
                cal_speed_delay_flag = 0
                break
        if cal_speed_delay_flag == 0:
            cal_speed_delay_flag = 1
            # print('jump')
            continue
        all_data[frame_now].append('speed')



# 遍历all_data
for line in all_data:
    print(line)

with open('gallery/' + person + '_predict.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(all_data)