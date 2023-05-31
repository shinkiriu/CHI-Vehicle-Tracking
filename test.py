#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:58:10 2023

@author: lijianhao
"""

# 车辆目标跟踪，使用yolov4方法
import cv2
import numpy as np
import time
import math
from object_detection import ObjectDetection  # 导入定义好的目标检测方法

filepath = "/" #define the path by yourself

od = ObjectDetection()

vehicle_list_frame = [] # vehicle_list_frame[0] means vehicle_list at frame 0

count = 0 # frame count

dict_class = {}

#（1）import video
cap = cv2.VideoCapture(filepath+'car_day.mp4')


#（2）export processed video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filepath+'car_day_detection.avi',fourcc, 20.0, (1920,1080),True)

file = open(filepath+'dnn_model/classes.txt','r')
 

# (3）Create a dict to store the key-value pairs(key: class_id, value: class_name)
k = 0
for line in file.readlines():
    v = line.strip()
    dict_class[k] = v
    k+=1

file.close()

print(dict_class)



# (4) Start reading video
while True:
    count += 1  # current frame number
    print('------------------------')
    print('Frame:', count)

    success, img = cap.read()
    
    if success == False:
        break
    
    # 返回class_ids图像属于哪个分类；scores图像属于某个分类的概率；boxes目标检测的识别框
    class_ids, scores, boxes  = od.detect(img)
    
    #vehicle_list at current frame, vehicle_list[0] means 0th vehicle
    vehicle_list = [] 
    
    print(len(boxes))
    
    # (5) Iterate each bounding box
    for i in range(len(boxes)):

        class_id = class_ids[i]
        score = scores[i]
        box = boxes[i]


        (x, y, w, h) = box
        
        # 获取每一个框的中心点坐标
        cx, cy = int((x+x+w)/2), int((y+y+h)/2) 

        vehicle_list.append([dict_class[class_id], [cx, cy]]) 
        
        # draw bounding box
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)  

        cv2.putText(img, str(dict_class[class_id])+":"+str(score*100)+"%", (x, y+h), 0, 1, (0,0,255),2)
    
       
    out.write(img)
    vehicle_list_frame.append(vehicle_list)
    
    
    if cv2.waitKey(1)==27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

