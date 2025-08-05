#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量人脸检测脚本
使用YOLOv9模型对数据集中的图片进行人脸检测，并将检测到的人脸坐标添加到对应的标签文件中
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import sys
import torch

# 添加yolov9目录到路径
yolov9_path = Path(__file__).parent / 'yolov9'
sys.path.append(str(yolov9_path))

from yolov9.models.common import DetectMultiBackend
from yolov9.utils.general import (check_img_size, non_max_suppression, scale_boxes, xyxy2xywh)
from yolov9.utils.augmentations import letterbox
from yolov9.utils.torch_utils import select_device

def convert_xyxy_to_yolo(xyxy, img_width, img_height):
    """
    将xyxy格式的坐标转换为YOLO格式 (x_center, y_center, width, height)
    """
    x1, y1, x2, y2 = xyxy
    
    # 计算中心点和宽高
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # 归一化到0-1范围
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return [x_center, y_center, width, height]

class FaceDetector:
    def __init__(self, weights_path, device='', img_size=640, conf_thres=0.25, iou_thres=0.45):
        self.device = select_device(device)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 加载模型
        print(f"正在加载模型: {weights_path}")
        self.model = DetectMultiBackend(weights_path, device=self.device)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.img_size = check_img_size((self.img_size, self.img_size), s=self.stride)
        
        # 预热模型
        self.model.warmup(imgsz=(1, 3, *self.img_size))
        print("模型加载完成")
    
    def detect_faces(self, img):
        """
        检测图片中的人脸
        """
        # 预处理图片
        im = letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        
        # 转换为tensor
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        
        # 推理
        pred = self.model(im, augment=False, visualize=False)
        
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=1000)
        
        results = []
        for i, det in enumerate(pred):
            if len(det):
                # 将检测框缩放回原图尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                
                # 提取坐标
                for *xyxy, conf, cls in reversed(det):
                    xyxy_list = [int(t) for t in xyxy]
                    results.append(xyxy_list)
        
        return results

def process_dataset(data_dir, model_weights, conf_threshold=0.25):
    """
    处理数据集中的所有图片
    """
    # 初始化人脸检测器
    detector = FaceDetector(model_weights, conf_thres=conf_threshold)
    
    # 处理train、test、valid三个目录
    for split in ['train', 'test', 'valid']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"警告: 目录 {split_dir} 不存在，跳过")
            continue
            
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"警告: 图片目录 {images_dir} 不存在，跳过")
            continue
            
        print(f"\n处理 {split} 数据集...")
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        total_images = len(image_files)
        processed_count = 0
        
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            
            # 对应的标签文件路径
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            print(f"处理图片: {img_file} ({processed_count + 1}/{total_images})")
            
            try:
                # 读取图片
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  警告: 无法读取图片 {img_path}")
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # 进行人脸检测
                detections = detector.detect_faces(img)
                
                # 读取现有的标签文件
                existing_labels = []
                if os.path.exists(label_path):
                    with open(label_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    existing_labels.append(parts)
                
                # 处理检测到的人脸
                new_face_labels = []
                for detection in detections:
                    # 转换坐标格式
                    yolo_coords = convert_xyxy_to_yolo(detection, img_width, img_height)
                    
                    # 创建标签行 (类别5 + 坐标)
                    label_line = ['5'] + [f"{coord:.6f}" for coord in yolo_coords]
                    new_face_labels.append(label_line)
                
                # 合并现有标签和新的人脸标签
                all_labels = existing_labels + new_face_labels
                
                # 写入更新后的标签文件
                with open(label_path, 'w', encoding='utf-8') as f:
                    for label in all_labels:
                        f.write(' '.join(label) + '\n')
                
                if new_face_labels:
                    print(f"  检测到 {len(new_face_labels)} 个人脸，已添加到标签文件")
                else:
                    print(f"  未检测到人脸")
                    
            except Exception as e:
                print(f"  错误: 处理图片 {img_file} 时出错: {str(e)}")
                continue
            
            processed_count += 1
        
        print(f"完成处理 {split} 数据集，共处理 {processed_count}/{total_images} 张图片")

def main():
    parser = argparse.ArgumentParser(description='批量人脸检测并更新标签文件')
    parser.add_argument('--data-dir', type=str, default='/data/txf/1/data',
                       help='数据集根目录路径')
    parser.add_argument('--weights', type=str, default='/data/txf/1/yolov9-face-detection-main/yolov9/yolov9.pt',
                       help='YOLOv9模型权重文件路径')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--device', type=str, default='',
                       help='设备选择 (cpu, 0, 1, 2, 3...)')
    
    args = parser.parse_args()
    
    print("批量人脸检测脚本")
    print(f"数据集目录: {args.data_dir}")
    print(f"模型权重: {args.weights}")
    print(f"置信度阈值: {args.conf_threshold}")
    print(f"设备: {args.device if args.device else '自动选择'}")
    
    # 检查数据集目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据集目录 {args.data_dir} 不存在")
        return
    
    # 检查模型文件是否存在
    if not os.path.exists(args.weights):
        print(f"错误: 模型权重文件 {args.weights} 不存在")
        return
    
    # 处理数据集
    process_dataset(args.data_dir, args.weights, args.conf_threshold)
    
    print("\n批量人脸检测完成！")

if __name__ == "__main__":
    main() 