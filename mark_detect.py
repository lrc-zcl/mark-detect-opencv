import cv2
import time
import numpy as np
import tensorflow as tf
from new_speech import tts_to_wav
import os
from playsound import playsound

"""
TTS功能
"""


class MaskDetection:

    def __init__(self, mode='rasp'):
        """
        加载人脸检测模型 和 口罩模型
        """
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
            tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存用量按需使用
            tf.config.set_visible_devices([gpu0], "GPU")

        self.mask_model = tf.keras.models.load_model('./model weights/mark_1.h5')
        # 类别标签
        self.labels = ['正常', '未佩戴', '不规范']
        # 标签对应颜色，BGR顺序，绿色、红色、黄色
        self.colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255)]

        # 获取label显示的图像
        self.zh_label_img_list = self.getLabelPngList()

    def getLabelPngList(self):
        """
        获取本地label显示的图像的列表
        """
        if os.path.exists('./speech_data/nomark.wav'):
            os.remove('./speech_data/nomark.wav')

        overlay_list = []
        for i in range(3):
            fileName = './markdataset/label_img/%s.png' % (i)
            overlay = cv2.imread(fileName, cv2.COLOR_RGB2BGR)
            overlay = cv2.resize(overlay, (0, 0), fx=0.3, fy=0.3)
            overlay_list.append(overlay)  # 标签列表的  标签图片的宽度和高度
        return overlay_list

    def imageBlob(self, face_region):
        """
        将图像转为blob
        """

        if face_region is not None:
            blob = cv2.dnn.blobFromImage(face_region, 1, (100, 100), (104, 117, 123), swapRB=True)
            blob_squeeze = np.squeeze(blob).T
            blob_rotate = cv2.rotate(blob_squeeze, cv2.ROTATE_90_CLOCKWISE)
            blob_flip = cv2.flip(blob_rotate, 1)
            # 对于图像一般不用附属，所以将它移除
            # 归一化处理
            blob_norm = np.maximum(blob_flip, 0) / blob_flip.max()
            return blob_norm
        else:
            return None

    def detect(self):
        """
        识别
        """
        face_detector = cv2.dnn.readNetFromCaffe("./markdataset/weights/deploy.prototxt.txt",
                                                 './markdataset/weights/res10_300x300_ssd_iter_140000.caffemodel')

        # face_detector = cv2.CascadeClassifier(
        # "D:/APPLICATION/anaconda/data/envs/opencv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")

        cap = cv2.VideoCapture(0)

        # 获取视频帧的高度和宽度
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frameTime = time.time()

        videoWriter = cv2.VideoWriter('./mark_result/' + str(time.time()) + '.mp4', cv2.VideoWriter_fourcc(*'H264'),
                                      10, (960, 720))

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            frame_resize = cv2.resize(frame, (300, 300))

            img_blob = cv2.dnn.blobFromImage(frame_resize, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True)  # 数据转换
            face_detector.setInput(img_blob)
            detections = face_detector.forward()
            num_of_detections = detections.shape[2]
            print("num_of_detections是{}".format(num_of_detections))
            # 记录人数(框)
            person_count = 0

            # 遍历多个
            for index in range(num_of_detections):
                # 置信度
                detection_confidence = detections[0, 0, index, 2]
                # 挑选置信度
                if detection_confidence > 0.5:

                    person_count += 1

                    # 位置坐标 记得放大 识别框人脸的位置
                    locations = detections[0, 0, index, 3:7] * np.array([frame_w, frame_h, frame_w, frame_h])  # 坐标比例放大
                    l, t, r, b = locations.astype('int')
                    # 裁剪人脸区域
                    face_region = frame[t:b, l:r]
                    # 转为blob格式 尺度也进行了变换
                    blob_norm = self.imageBlob(face_region)

                    if blob_norm is not None:
                        # 模型预测
                        img_input = blob_norm.reshape(1, 100, 100, 3)
                        result = self.mask_model.predict(img_input)

                        # softmax分类器处理
                        result = tf.nn.softmax(result[0]).numpy()

                        # 最大值索引
                        max_index = result.argmax()
                        # 最大值
                        max_value = result[max_index]
                        # 标签

                        label = self.labels[max_index]

                        # 开始语音录制 进行tts
                        if label == "未佩戴":
                            output_pcm_path_no = "./speech_data/nomark.pcm"
                            output_wav_path_no = "./speech_data/nomark.wav"
                            if os.path.exists(output_wav_path_no):
                                pass
                            else:
                                tts_to_wav(input_text="没有佩戴口罩，请佩戴口罩", output_pcm_path=output_pcm_path_no,
                                           output_wav_path=output_wav_path_no)
                                playsound(output_wav_path_no)
                        elif label == "不规范":
                            output_pcm_path_ff = "./speech_data/ffmark.pcm"
                            output_wav_path_ff = "./speech_data/ffmark.wav"
                            if os.path.exists(output_wav_path_ff):
                                pass
                            else:
                                tts_to_wav(input_text="没有正确佩戴口罩，请正确佩戴口罩", output_pcm_path=output_pcm_path_ff,
                                           output_wav_path=output_wav_path_ff)

                        # 对应中文标签
                        overlay = self.zh_label_img_list[max_index]
                        overlay_h, overlay_w = overlay.shape[:2]

                        # 覆盖范围
                        overlay_l, overlay_t = l, (t - overlay_h - 20)
                        overlay_r, overlay_b = (l + overlay_w), (overlay_t + overlay_h)

                        # 判断边界
                        if overlay_t > 0 and overlay_r < frame_w:
                            overlay_copy = cv2.addWeighted(frame[overlay_t:overlay_b, overlay_l:overlay_r], 1, overlay,
                                                           20, 0)
                            frame[overlay_t:overlay_b, overlay_l:overlay_r] = overlay_copy

                            cv2.putText(frame, str(round(max_value * 100, 2)) + "%", (overlay_r + 20, overlay_t + 40),
                                        cv2.FONT_ITALIC, 0.8, self.colors[max_index], 2)

                    # 人脸框
                    cv2.rectangle(frame, (l, t), (r, b), self.colors[max_index], 5)

            now = time.time()
            fpsText = 1 / (now - frameTime)
            frameTime = now

            #cv2.putText(frame, "FPS:  " + str(round(fpsText, 2)), (20, 40), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Person:  " + str(person_count), (20, 40), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 2)

            videoWriter.write(frame)
            cv2.imshow('Mask detect', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        videoWriter.release()
        cap.release()
        cv2.destroyAllWindows()


mask_detection = MaskDetection()
mask_detection.detect()
