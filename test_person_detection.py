#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8人体检测功能测试脚本
"""

import os
import sys
from pathlib import Path

def test_imports():
    """测试导入是否正常"""
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
        print("✓ 所有必要的模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_model_loading():
    """测试模型加载"""
    try:
        from ultralytics import YOLO
        
        weights_path = 'yolov8-detect/best.pt'
        if not os.path.exists(weights_path):
            print(f"✗ 模型文件不存在: {weights_path}")
            return False
        
        model = YOLO(weights_path)
        print("✓ 模型加载成功")
        return True
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False

def test_data_directory():
    """测试数据集目录"""
    data_dir = '/data/txf/1/data'
    if not os.path.exists(data_dir):
        print(f"✗ 数据集目录不存在: {data_dir}")
        return False
    
    print(f"✓ 数据集目录存在: {data_dir}")
    
    # 检查子目录
    for split in ['train', 'test', 'valid']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            images_dir = os.path.join(split_dir, 'images')
            labels_dir = os.path.join(split_dir, 'labels')
            
            if os.path.exists(images_dir):
                image_count = len([f for f in os.listdir(images_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                print(f"  ✓ {split}/images: {image_count} 张图片")
            else:
                print(f"  ✗ {split}/images: 目录不存在")
            
            if os.path.exists(labels_dir):
                label_count = len([f for f in os.listdir(labels_dir) 
                                 if f.lower().endswith('.txt')])
                print(f"  ✓ {split}/labels: {label_count} 个标签文件")
            else:
                print(f"  ✗ {split}/labels: 目录不存在")
        else:
            print(f"  ✗ {split}: 目录不存在")
    
    return True

def test_sample_detection():
    """测试样本检测"""
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
        
        # 检查是否有测试图片
        test_image_path = 'yolov8-detect/input.jpg'
        if not os.path.exists(test_image_path):
            print(f"✗ 测试图片不存在: {test_image_path}")
            return False
        
        # 加载模型
        model = YOLO('yolov8-detect/best.pt')
        
        # 读取测试图片
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"✗ 无法读取测试图片: {test_image_path}")
            return False
        
        # 进行检测
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, conf=0.5)
        
        # 检查检测结果
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.data.cpu().numpy()
            print(f"✓ 检测成功，检测到 {len(boxes)} 个目标")
            
            # 显示类别信息
            classes = set()
            for box in boxes:
                cls = int(box[5])
                classes.add(cls)
            print(f"  检测到的类别: {sorted(classes)}")
        else:
            print("✓ 检测成功，但未检测到目标")
        
        return True
    except Exception as e:
        print(f"✗ 样本检测失败: {e}")
        return False

def main():
    print("YOLOv8人体检测功能测试")
    print("=" * 50)
    
    # 测试导入
    print("\n1. 测试模块导入...")
    if not test_imports():
        return
    
    # 测试模型加载
    print("\n2. 测试模型加载...")
    if not test_model_loading():
        return
    
    # 测试数据集目录
    print("\n3. 测试数据集目录...")
    if not test_data_directory():
        return
    
    # 测试样本检测
    print("\n4. 测试样本检测...")
    if not test_sample_detection():
        return
    
    print("\n" + "=" * 50)
    print("✓ 所有测试通过！可以运行批量人体检测脚本")
    print("\n运行命令:")
    print("python3 batch_person_detection.py")

if __name__ == "__main__":
    main() 