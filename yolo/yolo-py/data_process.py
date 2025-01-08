import cv2
import numpy as np
import yaml

class YOLO_DATA_PROCESS:

    def __init__(self,classes,input_size=(640,640), confidence_thres=0.25, iou_thres=0.45):
        """
        初始化YOLOv8类的实例。

        参数:
            input_size:输入图片的大小
            classes:检测类别的标签路径，yaml格式，names属性 dict
            confidence_thres: 用于过滤检测结果的置信度阈值。
            iou_thres: 非极大值抑制的IoU（交并比）阈值。
        """
        with open(classes, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)  # 使用 safe_load 解析 YAML 文件
        if 'names' in data:  # 检查是否存在 names 属性
            self.classes = data['names']
        else:
            raise KeyError("The 'names' attribute does not exist in the YAML file.")

        self.confidence_thres = confidence_thres
        self.img = np.zeros((input_size[0],input_size[1],3))
        self.iou_thres = iou_thres
        self.input_size = input_size
        # 为每个类别生成一个颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        在输入图像上绘制检测到的边界框和标签。

        参数:
            img: 要在其上绘制检测结果的输入图像。
            box: 检测到的边界框。
            score: 对应的检测置信度分数。
            class_id: 检测到的对象的类别ID。

        返回值:
            无
        """
        # 提取边界框的坐标
        x1, y1, w, h = box

        # 获取类别ID对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 创建包含类别名称和置信度分数的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # 绘制填充矩形作为标签文本的背景
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # 在图像上绘制标签文本
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    def preprocess(self,img_path):
        """
        在执行推理之前预处理输入图像。

        返回值:
            image_data: 预处理后的图像数据，准备进行推理。
        """
        # 使用OpenCV读取输入图像
        self.img = cv2.imread(img_path)

        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]

        # 将图像的颜色空间从BGR转换为RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # 调整图像大小以匹配输入形状
        img = cv2.resize(img, self.input_size)

        # 通过除以255.0来规范化图像数据
        image_data = np.array(img)# / 255.0

        # 转置图像，使通道维度为第一个维度
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先

        # 扩展图像数据的维度以匹配预期的输入形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # 返回预处理后的图像数据
        return image_data
    
    def postprocess(self, output):
        """
        对模型的输出进行后处理，以提取边界框、置信度分数和类别ID。

        参数:
            input_image (numpy.ndarray): 输入图像。
            output (numpy.ndarray): 模型的输出。

        返回值:
            numpy.ndarray: 带有绘制检测结果的输入图像。
        """
        # 转置并压缩输出以匹配预期的形状   output size:8400*14
        outputs = np.transpose(np.squeeze(output[0]))
        # 获取输出数组中的行数
        rows = outputs.shape[0]  #8400

        # 用于存储检测到的边界框、置信度分数和类别ID的列表
        boxes = []
        scores = []
        class_ids = []

        # 计算边界框坐标的缩放因子
        x_factor = self.img_width / self.input_size[0]
        y_factor = self.img_height / self.input_size[1]

        # 遍历输出数组中的每一行
        for i in range(rows):
            # 从当前行中提取类别分数
            classes_scores = outputs[i][4:]

            # 找到类别分数中的最大值
            max_score = np.amax(classes_scores)

            # 如果最大值大于置信度阈值
            if max_score >= self.confidence_thres:
                # 获取具有最高分数的类别ID
                class_id = np.argmax(classes_scores)

                # 从当前行中提取边界框坐标
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 计算缩放后的边界框坐标
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # 将类别ID、置信度分数和边界框坐标添加到各自的列表中
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # 应用非极大值抑制以过滤重叠的边界框
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        img_predict = self.img
        sum_box = []
        # 遍历非极大值抑制后选择的索引
        for i in indices:
            # 获取对应于索引的边界框、置信度分数和类别ID
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            #sum_box.append(box)
            # 在输入图像上绘制检测结果
        
            self.draw_detections(img_predict, box, score, class_id)
        #print(f"sum_box:{sum_box}")
        # 返回修改后的输入图像
        return img_predict  