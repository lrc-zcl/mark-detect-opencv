#  <p align="center">口罩检测系统</p>
<p style="text-indent: 2em;">该项目基于 OpenCV 和 TensorFlow 实现口罩检测系统。首先通过 OpenCV 对人脸进行检测，然后将检测到的人脸切分成 100x100 尺寸大小的图片。接着，将这些图片送入 CNN 神经网络模型进行推理，以实现口罩的检测。在整个过程中，项目还利用 WebSocket 调用科大讯飞的在线语音识别系统，用于播报识别结果。</p>
