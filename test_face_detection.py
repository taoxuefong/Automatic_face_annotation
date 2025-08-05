#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试人脸检测功能
"""

import os
import sys
from pathlib import Path

# 添加yolov9目录到路径
yolov9_path = Path(__file__).parent / 'yolov9'
sys.path.append(str(yolov9_path))

def test_imports():
    """测试导入是否正常"""
    try:
        from yolov9.models.common import DetectMultiBackend
        from yolov9.utils.general import (check_img_size, non_max_suppression, scale_boxes)
        from yolov9.utils.augmentations import letterbox
        from yolov9.utils.torch_utils import select_device
        print("✓ 所有必要的模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_model_loading():
    """测试模型加载"""
    try:
        from yolov9.models.common import DetectMultiBackend
        from yolov9.utils.torch_utils import select_device
        
        weights_path = '/data/txf/1/yolov9-face-detection-main/yolov9/yolov9.pt'
        if not os.path.exists(weights_path):
            print(f"✗ 模型文件不存在: {weights_path}")
            return False
        
        device = select_device('')
        model = DetectMultiBackend(weights_path, device=device)
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

def main():
    print("人脸检测功能测试")
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
    
    print("\n" + "=" * 50)
    print("✓ 所有测试通过！可以运行批量人脸检测脚本")
    print("\n运行命令:")
    print("python batch_face_detection.py")

if __name__ == "__main__":
    main() 