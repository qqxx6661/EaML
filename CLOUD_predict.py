#!/usr/bin/env python3
# coding=utf-8
import csv


def relative_position(cood):
    # 输入两点坐标的list，其实就是矩形的左上角和右下角坐标。求该点在画面内的相对位置（左右来看）。视频分辨率为640x480。
    # 举例：该点x坐标是320，那他的相对位置就是0
    value = ((0.5 * (cood[0] + cood[2])) - 320) / 320
    value = int(value * 100)
    return value


def cam_predict_relative(cam_id, position):  # 由于场景是预先设计好的，所以这里需要手动设置
    cam = [0, 0, 0, 0, 0, 0]
    cam[cam_id] = 100
    if cam_id == 0:
        if position > 0: cam[1] = position
    elif cam_id == 1:
        if position > 0: cam[2] = position
        else: cam[0] = abs(position)
    elif cam_id == 2:
        if position > 0: cam[3] = position
        else: cam[1] = cam[5] = abs(position)
    elif cam_id == 3:
        if position < 0: cam[2] = abs(position)
    elif cam_id == 4:
        if position < 0: cam[5] = abs(position)
    else:
        if position > 0: cam[4] = position
        else: cam[2] = abs(position)
    return cam


def judge_cam_location(curr_line, prev_list):  # 判断是否关联并且进入正负数是否合理
    if prev_list[1] == 0:
        if curr_line[1] == 1 and curr_line[3] < 0 and prev_list[3] > 0: return True
    if prev_list[1] == 1:
        if curr_line[1] == 0 and curr_line[3] > 0 and prev_list[3] < 0: return True
        if curr_line[1] == 2 and curr_line[3] < 0 and prev_list[3] > 0: return True
    if prev_list[1] == 2:
        if curr_line[1] == 1 and curr_line[3] > 0 and prev_list[3] < 0: return True
        if curr_line[1] == 3 and curr_line[3] < 0 and prev_list[3] > 0: return True
        if curr_line[1] == 5 and curr_line[3] < 0 and prev_list[3] < 0: return True  # 2到5比较特殊
    if prev_list[1] == 3:
        if curr_line[1] == 2 and curr_line[3] > 0: return True  # 1547的3中回来并未识别出来，特殊照顾
    if prev_list[1] == 5:
        if curr_line[1] == 2 and curr_line[3] > 0 and prev_list[3] < 0: return True
        if curr_line[1] == 4 and curr_line[3] < 0 and prev_list[3] > 0: return True
    if prev_list[1] == 4:
        if curr_line[1] == 5 and curr_line[3] > 0 and prev_list[3] < 0: return True
    return False

exp_info = 'yolo_1547_'
person = 'person_1'  # 文件名
cal_speed_delay = 6  # 连续在同一摄像头n帧后再计算速度
cal_speed_delay_flag = 1  # 连续在同一摄像头n帧都有数据则置为1
# 创建所有帧数组
all_data = []
for frame in range(1800):
    all_data.append([frame, ])
all_data_ML = []

# 读取
with open('gallery/' + exp_info + person + '.csv') as csvFile:
    reader = csv.reader(csvFile)
    for item in reader:
        frame_now = int(item[0])  # 当前处理帧

        # 长度大于1说明这帧之前有了，异常.由于暂时一共就两个人，所以只比较两个值，选择距离比较小的一个
        if len(all_data[frame_now]) > 1:
            # 这里若报错，说明前面一帧也没有信息，假设没有出现这种情况
            last_position = int(all_data[frame_now-1][3])
            if abs(all_data[frame_now][3] - last_position) > abs(relative_position(eval(item[2])) - last_position):
                while len(all_data[frame_now]) > 1:
                    all_data[frame_now].pop()
            else:
                # 不需要存储了，使用上次数据
                continue

        all_data[frame_now].append(int(item[1]))  # 加入camid
        all_data[frame_now].append(eval(item[2]))  # 加入位置，主要用于速度计算
        all_data[frame_now].append(relative_position(eval(item[2])))  # 加入相对位置
        # 加入速度
        for i in range(cal_speed_delay):
            # 连续N帧有
            if len(all_data[frame_now - (i+1)]) == 1:
                cal_speed_delay_flag = 0
                break
            # 连续N帧在同一个摄像头内
            if all_data[frame_now - (i+1)][1] != item[1]:
                cal_speed_delay_flag = 0
                break
        if cal_speed_delay_flag == 0:
            cal_speed_delay_flag = 1
            # print('jump')
            continue
        speed_x = int(eval(item[2])[0]) - int(all_data[frame_now-1][2][0])
        speed_y = int(eval(item[2])[1]) - int(all_data[frame_now-1][2][1])
        all_data[frame_now].append([speed_x, speed_y])  # x,y轴速度

# 遍历all_data
# for line in all_data:
#     print(line)


# 这里插入优化函数，去除出错的信息，之后再存入person_x_predict中
prev_frame = []
for l, line in enumerate(all_data):
    if len(line) == 1:  # 如果该帧没信息，跳过
        continue
    else:
        if not prev_frame:  # 如果之前没信息，给第一次信息，跳过
            prev_frame = line
        else:
            if line[1] == prev_frame[1]:  # 如果相同摄像头,距离绝对值必须小于50
                if abs(line[3] - prev_frame[3]) <= 50:
                    prev_frame = line
                else:
                    print('去除', line, '对比', prev_frame)
                    all_data[l] = line[:1]
            else:  # 如果摄像头不相同，必须满足1.相连摄像头，2.距离一左一右正负必须相反
                if judge_cam_location(line, prev_frame):
                    prev_frame = line
                else:
                    print('去除', line, '对比', prev_frame)
                    all_data[l] = line[:1]



# 写入person_x_predict
with open('gallery/' + exp_info + person + '_predict.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(all_data)

# 写入person_x_ML
for line in all_data:
    if len(line) == 5:  # 有速度的才处理
        ML_temp = []
        ML_temp.append(line[1])
        ML_temp.append(line[0])
        ML_temp.append(line[4][0])
        ML_temp.append(line[4][1])
        cam_list = cam_predict_relative(int(line[1]), line[3])
        for cam_value in cam_list:
            ML_temp.append(cam_value)
        all_data_ML.append(ML_temp)

with open('gallery/' + exp_info + person + '_ML.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(all_data_ML)
