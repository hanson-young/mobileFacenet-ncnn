import cv2
import os
import random

root_path = '/media/hanson/0C5419E40C5419E4/BaiduNetdiskDownload/china/400W'

path_list = os.listdir(root_path)
random.shuffle(path_list)
print(len(path_list))

for img_path in path_list:
    img_path = os.path.join(root_path, img_path)
    print(img_path.split('/')[-1])
    img = cv2.imread(img_path)
    cv2.imshow('img', img)
    cv2.waitKey(0)
