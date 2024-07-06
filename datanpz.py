"""
将人脸数据集进行预处理，压缩保存至npz
"""
import numpy as np
import os , glob
import tqdm
import cv2
from face_data_detect import face_detect , imgBlob
labels = os.listdir("./markdataset/images")

img_list = []
label_list = []
for label in labels:
    # 获取每类文件列表
    file_list = glob.glob('./markdataset/images/%s/*.jpg' % (label))

    for img_file in tqdm.tqdm(file_list, desc="处理文件夹 %s " % (label)):
        print(img_file)
        # 读取文件
        img = cv2.imread(img_file)
        img_crop = img
        # 裁剪人脸
        #img_crop = face_detect(img)
        # 转为Blob
        if img_crop is not None:
            img_blob = imgBlob(img_crop)
            img_list.append(img_blob)
            label_list.append(label)
X = np.asarray(img_list)
Y = np.asarray(label_list)
np.savez('./markdataset/datanpz/imageData.npz', X, Y)
