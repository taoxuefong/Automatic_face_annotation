from ultralytics import YOLO
import cv2
import time
import pandas as pd
# 加载预训练的 YOLOv8 模型
model = YOLO("best.pt")
def detect_person(image):
    image = cv2.imread(image_path)
    start_time = time.time()
    # 使用 YOLOv8 模型进行推理
    results = model(image,conf=0.5)
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
    # 显示带检测结果的帧
   
# 加载本地图片
image_path = 'input.jpg'  # 替换为您的图片路径
image = cv2.imread(image_path)
while True:
        image_path = input("请输入图像地址（输入 'exit' 退出）：").strip()
        if image_path.lower() == "exit":
            print("程序已退出。")
            break
        print(detect_person(image_path))

