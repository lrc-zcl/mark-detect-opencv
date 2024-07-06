import _thread as thread
import base64
import datetime
import hashlib
import hmac
import wave
import json
import os
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import time
import websocket

STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Text):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {"aue": "raw", "auf": "audio/L16;rate=16000", "vcn": "xiaoyan", "tte": "utf8"}
        self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode('utf-8')), "UTF8")}
        #使用小语种须使用以下方式，此处的unicode指的是 utf16小端的编码方式，即"UTF-16LE"”
        #self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode('utf-16')), "UTF8")}

    # 生成url
    def create_url(self):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url


def on_message(ws, message):
    try:
        message = json.loads(message)
        code = message["code"]
        sid = message["sid"]
        audio = message["data"]["audio"]
        audio = base64.b64decode(audio)
        status = message["data"]["status"]
        print(message)
        if status == 2:
            print("ws is closed")
            ws.close()
        if code != 0:
            errMsg = message["message"]
            print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))
        else:
            with open(ws.output_pcm, 'ab') as f:
                f.write(audio)

    except Exception as e:
        print("receive msg, but parse exception:", e)


def on_error(ws, error):
    print("### error:", error)


def on_close(ws, *args):  # Adjust on_close to accept variable arguments
    print("### closed ###")


def on_open(ws, wsParam):
    def run(*args):
        d = {"common": wsParam.CommonArgs,
             "business": wsParam.BusinessArgs,
             "data": wsParam.Data,
             }
        d = json.dumps(d)
        print("------>开始发送文本数据")
        ws.send(d)
        if os.path.exists(ws.output_pcm):
            os.remove(ws.output_pcm)

    thread.start_new_thread(run, ())


# def tts_to_wav(input_text, output_pcm_path, output_wav_path):
#     wsParam = Ws_Param(APPID='3bf3ea52', APISecret='MmYzZWJjZjQ5MWE1MzFmM2FhMDMxZTJm',
#                        APIKey='50392e468869ed4fdb1cae1dad31109c',
#                        Text=input_text)
#
#     ws = websocket.WebSocketApp(wsParam.create_url(),
#                                 on_message=lambda ws, msg: on_message(ws, msg),
#                                 on_error=on_error,
#                                 on_close=lambda ws, *args: on_close(ws, *args))
#
#     ws.output_pcm = output_pcm_path  # Pass output_pcm as an attribute to ws
#
#     def on_open_internal(*args):
#         on_open(ws, wsParam)
#
#     ws.on_open = on_open_internal
#     ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
#
#     print("pcm合成结束时间" + str(time.time()))
#     with open(output_pcm_path, 'rb') as pcmfile:
#         pcmdata = pcmfile.read()
#     with wave.open(output_wav_path, 'wb') as wavfile:
#         wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
#         wavfile.writeframes(pcmdata)
#     print("转wav结束时间" + str(time.time()))
import time

def tts_to_wav(input_text, output_pcm_path, output_wav_path):
    wsParam = Ws_Param(APPID='3bf3ea52', APISecret='MmYzZWJjZjQ5MWE1MzFmM2FhMDMxZTJm',
                       APIKey='50392e468869ed4fdb1cae1dad31109c',
                       Text=input_text)

    ws = websocket.WebSocketApp(wsParam.create_url(),
                                on_message=lambda ws, msg: on_message(ws, msg),
                                on_error=on_error,
                                on_close=lambda ws, *args: on_close(ws, *args))

    ws.output_pcm = output_pcm_path  # Pass output_pcm as an attribute to ws

    def on_open_internal(*args):
        on_open(ws, wsParam)

    ws.on_open = on_open_internal
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    print("pcm合成结束时间" + str(time.time()))

    with open(output_pcm_path, 'rb') as pcmfile:
        pcmdata = pcmfile.read()

    with wave.open(output_wav_path, 'wb') as wavfile:
        wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        wavfile.writeframes(pcmdata)

    print("转wav结束时间" + str(time.time()))

# 在 while 循环中调用 tts_to_wav，每次调用都使用新的 WAV 文件路径
# while True:
#     output_wav_path = f"output_{time.time()}.wav"
#     tts_to_wav("你的文本", "output.pcm", output_wav_path)

# Example usage




