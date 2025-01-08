from rknn.api import RKNN


#0-set your rknn config
MEAN_VALUES = [[0,0,0]]#[[127.5,127.5,127.5]]
STD_VALUES = [[255,255,255]]#[[127.5,127.5,127.5]]      
##目前支持的量化类型有w8a8、 w4a16、 w8a16、 w4a8、w16a16i和w16a16i_dfp。默认值为w8a8。
QUANT_DTYPE = "w8a8"
TARGET_PLATFORM = "rk3588"
##choose your model
MODEL_TYPE_LIST = ["Caffe","TensorFlow","TensorFlow Lite","ONNX","DarkNet","Pytorch"]
MODEL_TYPE = MODEL_TYPE_LIST[3]
INPUT_SIZE = [[1,3,640,640]]
MODEL_PATH = "./model/yolov11/200epoch-adam.onnx"
##build cfg
DO_QUANT = False
DATASET_PATH = "./subdata/subdata.txt"  #用于矫正量化的数据集，格式txt文件
##rknn model output path
OUTPUT_PATH = "./model/yolov11/200epoch-adam.rknn"



#1-init  showlog
rknn = RKNN(verbose=True)

#2-config
print('--> Config model')
rknn.config(mean_values=MEAN_VALUES,std_values=STD_VALUES,quantized_dtype=QUANT_DTYPE,target_platform=TARGET_PLATFORM)
print('done')

#3-load your model
print('--> Loading model')
if(MODEL_TYPE_LIST[0] == MODEL_TYPE):
    rknn.load_caffe(model=MODEL_PATH)
elif(MODEL_TYPE_LIST[1] == MODEL_TYPE):
    rknn.load_tensorflow(tf_pb=MODEL_PATH)
elif(MODEL_TYPE_LIST[2] == MODEL_TYPE):
    rknn.load_tflite(model=MODEL_PATH)
elif(MODEL_TYPE_LIST[3] == MODEL_TYPE):
    rknn.load_onnx(model=MODEL_PATH)
elif(MODEL_TYPE_LIST[4] == MODEL_TYPE):
    rknn.load_darknet(model=MODEL_PATH)
elif(MODEL_TYPE_LIST[5] == MODEL_TYPE):
    rknn.load_pytorch(model=MODEL_PATH,input_size_list=INPUT_SIZE)
else:
    print("model input error!")
    exit(1)
print("done")

#4-bulid your model
print('--> Building model')
ret = rknn.build(do_quantization=DO_QUANT, dataset=DATASET_PATH)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')


#5-Export rknn model
print('--> Export rknn model')
ret = rknn.export_rknn(OUTPUT_PATH)
if ret != 0:
    print('Export rknn model failed!')
    exit(ret)
print('done')

#release
rknn.release()