import wx
import os
import cv2
import core
import device
from utils import load_graph_model


def selectModel():
    path = os.path.dirname(os.path.abspath(__file__))
    number = 0
    if number == 0:
        modelPath = path+r"\Mobnet075F-model-stride16.json"
    return modelPath

def initializeModel(modelPath):
    graph = load_graph_model(modelPath)
    return graph
        
    
def loadCamera(id,width,height):
    cam_obj = cv2.VideoCapture(id,cv2.CAP_DSHOW)
    if cam_obj.isOpened()==False:
        cam_obj.open()
    cam_obj.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam_obj


def closeCamera(cam_obj):
    cam_obj.release()

def selectCamera(last_index):
    number = 0
    hint = "請選擇攝影機 (選項 0 到 " + str(last_index) + " ): "
    
    if last_index ==  0:
        return number

    try:
        number = int(input(hint))
        
    except Exception:
        print("輸入非數字，請重試!")
        return selectCamera(last_index)

    if number > last_index or number < 0:
        print("不存在的編號，請重試!")
        return selectCamera(last_index)

    return number

def selectResolution():
    number = 1
    print("0:1080p")
    print("1:720p")
    print("2:480p")
    hint = "請選擇攝影機解析度(選項 0 到 2 ):"

    try:
        number = int(input(hint))

    except Exception:
        print("輸入非數字，請重試!")
        return select_resolution()

    if number > 2 or number < 0:
        print("不存在的編號，請重試!")
        return selectResolution()

    elif number == 0:
        return 1920,1080

    elif number == 1:
        return 1280,720

    else:
        return 640,480

def selectBackground():
    hint = "是否自訂背景？(0:是 1:否)"
    number = 1

    try:
        number = int(input(hint))

    except Exception:
        print("輸入非數字，請重試!")
        return selectBackground()

    if number > 1 or number < 0:
        print("不存在的編號，請重試!")
        return selectBackground()

    elif number == 1:
        return False, None

    else:
        path = input("請輸入背景圖片完整路徑(可直接拖曳檔案至本視窗):")

        if os.path.isfile(path):
            return True, path

        else:
            print("找不到檔案，請確認路徑後重試!")
            return selectBackground()

def calc_targetResolution(stride,length):
    target = (int(length) // stride) * stride + 1
    return target



os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
deviceList = device.getDeviceList()
index = 0

for name in deviceList:
    print(str(index) + ': ' + name)
    index += 1
lastIndex = index - 1
if lastIndex < 0:
    print("No device is connected")



stride = 16 #add option for model in future
modelPath = selectModel()
graph = initializeModel(modelPath)
camId = selectCamera(lastIndex)

imageW, imageH = selectResolution()
targetW = calc_targetResolution(stride,imageW)
targetH = calc_targetResolution(stride,imageH)

useBackground, bgPath = selectBackground()

cap = loadCamera(camId, imageW, imageH)
ret, frame = cap.read()
while(ret):
    final=core.make_final(graph, frame, imageW, imageH, targetW, targetH, stride, bgPath, useBackground)
    cv2.imshow('Output', final)
    key = cv2.waitKey(1)
    if key == 27:
        break
    ret, frame = cap.read()
closeCamera(cap)