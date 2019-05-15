import cv2
import glob
import os

root = '/home/hanson/DataSet/face-recognition/child'
save = '/home/hanson/DataSet/face-recognition/child/child_china'
for path in glob.glob(os.path.join(root, '*.jpg')):
    id_folder = os.path.join(save, path.split('/')[-1].split('_')[0])
    if not os.path.exists(id_folder):
        os.makedirs(id_folder)
    img = cv2.imread(path)
    cv2.imwrite(os.path.join(id_folder, path.split('/')[-1]),img)