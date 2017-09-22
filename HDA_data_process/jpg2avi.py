import os
import cv2

img_root = '/home/zhendongyang/PycharmProjects/EaML/HDA_data_process/HDAimage/camera50/'#这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠
fps = 20    # 保存视频的FPS，可以适当调整

# 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('/home/zhendongyang/PycharmProjects/EaML/HDA_data_process/HDAimage/camera50.avi', fourcc, fps, (1280, 800))  # 最后一个是保存图片的尺寸

for i in range(2227):
    frame = cv2.imread(img_root+str(i+1)+'.jpg')
    print('正在处理：', i)
    videoWriter.write(frame)
videoWriter.release()
