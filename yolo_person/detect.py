import time
from ultralytics import YOLO
import cv2
import numpy as np
import ctypes
import pandas as pd
# 加载 YOLOv5 模型
# 加载预训练的 YOLOv8 模型
model = YOLO('best.pt').load  # 使用适当的模型权重文件，例如 yolov8n.pt

def read_data_from_address(address, shape):
    # 创建一个指针指向指定地址
    data_pointer = ctypes.cast(address, ctypes.POINTER(ctypes.c_uint8))
    # 读取数据
    data = np.ctypeslib.as_array(data_pointer, shape=shape)
    return data
def detect_person(address,shape):
    """
    检测图像中的人体并保存结果
    :param image_path: 输入图像路径
    :param output_path: 输出图像路径
    """
   
    # 开始计时
    start_time = time.time()
    img=read_data_from_address(address, shape)
    #img=Image.open(img_path)
    # 加载图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 推理
    start_time = time.time()
    # 使用 YOLOv8 模型进行推理
    results = model(img_rgb,conf=0.5)
    detections = results[0].boxes.data.cpu().numpy()
    end_time = time.time()
    if len(detections) == 0:
        print("未检测到任何目标。")
        return
    # 构建表格数据
    data = []
    for detection in detections:
        xmin, ymin, xmax, ymax, conf = detection[:5]
        data.append({
            "xmin": int(xmin),
            "ymin": int(ymin),
            "xmax": int(xmax),
            "ymax": int(ymax),
            "conf": round(float(conf), 2)
        })
    # 打印检测结果 
    print(pd.DataFrame(data))
    print(f"检测耗时：{end_time - start_time:.2f} 秒")
    return pd.DataFrame(data)

if __name__ == "__main__":
    # 输入图像路径
    image_shape=(640, 640, 3)  # 替换为您的输入图像类型
    address=0x7f8b4c000000  # 替换为您的输入图像内存地址
    #output_dir = "runs/detect"  # 输出结果保存路径
    while True:
        # 提示用户输入图像路径
        address = input("请输入图像内存地址（输入 'exit' 退出）：").strip()
        if address.lower() == "exit":
            print("程序已退出。")
            break
        data_array=detect_person(address,image_shape)
        print(data_array)
    