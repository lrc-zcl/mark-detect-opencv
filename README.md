#  <p align="center">口罩检测系统</p>
<div style="text-indent: 2em;">
  该项目基于opencv和tensorflow实现口罩检测系统。  
  首先通过opencv对人脸进行检测，然后切分成100x100尺寸大小的图片，最后送至CNN神经网络模型里进行推理，实现口罩的检测，该过程中还使用了websocket调用科大讯飞的在线语音识别系统，进行识别结果的播报。
