"""
通过opencv数据集进行人脸识别，并进行剪切
"""
import cv2
import numpy as np
import tqdm
import os, glob

face_detector = cv2.dnn.readNetFromCaffe("./markdataset/weights/deploy.prototxt.txt",
                                         './markdataset/weights/res10_300x300_ssd_iter_140000.caffemodel')


# 人脸检测函数
def face_detect(img):
    # 转为Blob
    img_blob = cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 177, 123), swapRB=True)
    # 输入
    face_detector.setInput(img_blob)
    # 推理
    detections = face_detector.forward()
    # 获取原图尺寸
    img_h, img_w = img.shape[:2]

    # 人脸框数量
    person_count = detections.shape[2]

    for face_index in range(person_count):
        # 通过置信度选择
        confidence = detections[0, 0, face_index, 2]
        if confidence > 0.5:
            locations = detections[0, 0, face_index, 3:7] * np.array([img_w, img_h, img_w, img_h])
            # 获得坐标 记得取整
            l, t, r, b = locations.astype('int')
            return img[t:b, l:r]
    return None


# 转为Blob格式函数
def imgBlob(img):
    # 转为Blob
    img_blob = cv2.dnn.blobFromImage(img, 1, (100, 100), (104, 177, 123), swapRB=True)
    # 维度压缩
    img_squeeze = np.squeeze(img_blob).T
    # 旋转
    img_rotate = cv2.rotate(img_squeeze, cv2.ROTATE_90_CLOCKWISE)
    # 镜像
    img_flip = cv2.flip(img_rotate, 1)
    # 去除负数，并归一化
    img_blob = np.maximum(img_flip, 0) / img_flip.max()
    return img_blob
