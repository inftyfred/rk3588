import torch
import torch.nn
#import onnx
 
model = torch.load('./model/yolov11/100epoch-adam.pt')
#model.eval()
 
input_names = ['input']
output_names = ['output']
 
x = torch.randn(1,3,640,640,requires_grad=True)
 
torch.onnx.export(model, x, './model/yolov11/100epoch-adam.onnx',input_names=input_names, output_names=output_names, verbose='True')