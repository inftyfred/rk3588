import os
import cv2
import numpy as np
from rknn.api import RKNN
from data_process import YOLO_DATA_PROCESS

#yolov8s ./model/test_no_quant.rknn 13fps

#yolov8n ./model/my_no_quant.rknn  27fps

#yolov11n ./model/100epoch-adam.rknn 25fps


#confg
RKNN_MODEL_PATH = "./model/yolov8/yolov8s_no_quant.rknn"#"./model/yolov11/200epoch-adam.rknn"
TARGET = 'rk3588'
TEST_DATA_PATH = "./test"
# 获取文件夹中的所有文件名
file_names = os.listdir(TEST_DATA_PATH)
# 过滤掉文件夹，只保留文件
test_data = [os.path.join(TEST_DATA_PATH, f) for f in file_names if os.path.isfile(os.path.join(TEST_DATA_PATH, f))]
NP_SAVE_PATH = "./output/yolov8/yolov8s_no_quant.npy"#"./output/yolov11/200epoch-adam.npy"
DATA_CLASSES_PATH = "./VisDrone.yaml"
INPUT_SIZE = (640,640) #w*h
#置信度
OBJ_THRESH = 0.3
NMS_THRESH = 0.7

ydp = YOLO_DATA_PROCESS(classes=DATA_CLASSES_PATH,input_size=INPUT_SIZE,confidence_thres=OBJ_THRESH,iou_thres=NMS_THRESH)


rknn = RKNN()
#load rknn model
ret = rknn.load_rknn(path=RKNN_MODEL_PATH)
if (ret != 0) :
    print("load rknn model error!")
    exit(ret)

#set input
img = test_data[10]#"./test/0000006_01275_d_0000004.jpg"#test_data[3]   #路径 "./bus.jpg"
img = ydp.preprocess(img) #np


# 初始化运行时环境
print('--> Init runtime environment')
ret = rknn.init_runtime(target=TARGET)
if(ret!=0):
    print("init rknn error")
    exit(ret)
else:
    print("done")


#inference
print('--> Running model')
outputs = rknn.inference(inputs=[img],data_format=['nchw'])#default:nhwc
perf_detail = rknn.eval_perf() #模型性能 str
mem_detail = rknn.eval_memory() #模型内存占用 dict
print(f"模型性能评估：\n {perf_detail}")
#print(f"模型内存占用：\n {mem_detail}")
np.save(NP_SAVE_PATH, outputs[0])
img_predict = ydp.postprocess(output=outputs)
cv2.imshow(winname="predict",mat=img_predict)
cv2.waitKey()
print('done')


rknn.release()