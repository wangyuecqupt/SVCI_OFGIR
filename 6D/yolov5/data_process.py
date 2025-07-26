import os
import cv2
import json

val_json = json.load(open(r'E:\yolo_v5\num_dataset\dataset\valid_label.json'))
for x in val_json:
    print(x)
    img = cv2.imread(r"E:\yolo_v5\num_dataset\dataset\images\valid\images/" + x)
    width = img.shape[1]
    height = img.shape[0]
    val_label = list(map(int, val_json[x]['label']))
    val_height = list(map(int, val_json[x]['height']))
    val_left = list(map(int, val_json[x]['left']))
    val_width = list(map(int, val_json[x]['width']))
    val_top = list(map(int, val_json[x]['top']))
    loc_pic = r"E:\yolo_v5\num_dataset\dataset\images\valid\labels/" + x.split('.')[0] + '.txt'
    pic = open(loc_pic, "w")
    for i in range(len(val_label)):
        pic_label = val_label[i]
        pic_x = (val_left[i] + val_width[i] / 2) / width
        pic_y = (val_top[i] + val_height[i] / 2) / height
        pic_width = val_width[i] / width
        pic_height = val_height[i] / height
        pic.write(str(pic_label) + " " + str(pic_x) + " " + str(pic_y) + " " + str(pic_width) + " " + str(pic_height))
        pic.write("\n")
    pic.close()
