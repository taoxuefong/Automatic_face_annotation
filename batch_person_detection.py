#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量人体检测脚本
使用YOLOv8模型对数据集中的图片进行人体检测，删除原有类别3标签并添加新检测的人体坐标
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import sys
from ultralytics import YOLO

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

class PersonDetector:
    def __init__(self, weights_path, conf_thres=0.5):
        self.conf_thres = conf_thres
        
        # 加载模型
        print(f"正在加载YOLOv8模型: {weights_path}")
        self.model = YOLO(weights_path)
        print("模型加载完成")
    
    def detect_persons(self, img):
        """
        检测图片中的人体
        """
        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 推理
        results = self.model(img_rgb, conf=self.conf_thres)
        
        # 提取检测结果
        detections = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box[:6]
                # 保留所有检测结果（因为这是专门的人体检测模型）
                detections.append([int(x1), int(y1), int(x2), int(y2)])
        
        return detections

def filter_labels_by_class(labels, exclude_class=3):
    """
    过滤标签，删除指定类别的标签
    """
    filtered_labels = []
    for label in labels:
        if len(label) >= 5:
            class_id = int(label[0])
            if class_id != exclude_class:
                filtered_labels.append(label)
    return filtered_labels

def process_dataset(data_dir, model_weights, conf_threshold=0.5):
    """
    处理数据集中的所有图片
    """
    # 初始化人体检测器
    detector = PersonDetector(model_weights, conf_thres=conf_threshold)
    
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
                
                # 删除原有类别3的标签
                original_count = len([label for label in existing_labels if int(label[0]) == 3])
                filtered_labels = filter_labels_by_class(existing_labels, exclude_class=3)
                removed_count = original_count - len([label for label in filtered_labels if int(label[0]) == 3])
                
                # 进行人体检测
                detections = detector.detect_persons(img)
                
                # 处理检测到的人体
                new_person_labels = []
                for detection in detections:
                    # 转换坐标格式
                    yolo_coords = convert_xyxy_to_yolo(detection, img_width, img_height)
                    
                    # 创建标签行 (类别3 + 坐标)
                    label_line = ['3'] + [f"{coord:.6f}" for coord in yolo_coords]
                    new_person_labels.append(label_line)
                
                # 合并过滤后的标签和新的人体标签
                all_labels = filtered_labels + new_person_labels
                
                # 写入更新后的标签文件
                with open(label_path, 'w', encoding='utf-8') as f:
                    for label in all_labels:
                        f.write(' '.join(label) + '\n')
                
                # 输出处理结果
                if removed_count > 0:
                    print(f"  删除了 {removed_count} 个原有的人体标签")
                if new_person_labels:
                    print(f"  检测到 {len(new_person_labels)} 个人体，已添加到标签文件")
                else:
                    print(f"  未检测到人体")
                    
            except Exception as e:
                print(f"  错误: 处理图片 {img_file} 时出错: {str(e)}")
                continue
            
            processed_count += 1
        
        print(f"完成处理 {split} 数据集，共处理 {processed_count}/{total_images} 张图片")

def main():
    parser = argparse.ArgumentParser(description='批量人体检测并更新标签文件')
    parser.add_argument('--data-dir', type=str, default='/data/txf/1/data',
                       help='数据集根目录路径')
    parser.add_argument('--weights', type=str, default='yolov8-detect/best.pt',
                       help='YOLOv8模型权重文件路径')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='置信度阈值')
    
    args = parser.parse_args()
    
    print("批量人体检测脚本")
    print(f"数据集目录: {args.data_dir}")
    print(f"模型权重: {args.weights}")
    print(f"置信度阈值: {args.conf_threshold}")
    print("注意: 将删除原有类别3标签，并添加新检测的人体坐标")
    
    # 检查数据集目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据集目录 {args.data_dir} 不存在")
        return
    
    # 检查模型文件是否存在
    if not os.path.exists(args.weights):
        print(f"错误: 模型权重文件 {args.weights} 不存在")
        return
    
    # 确认操作
    print("\n警告: 此操作将删除所有原有类别3的标签，并替换为新检测的人体坐标")
    confirm = input("是否继续？(y/N): ").strip().lower()
    if confirm != 'y':
        print("操作已取消")
        return
    
    # 处理数据集
    process_dataset(args.data_dir, args.weights, args.conf_threshold)
    
    print("\n批量人体检测完成！")

if __name__ == "__main__":
    main() 