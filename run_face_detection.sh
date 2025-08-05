#!/bin/bash

# 批量人脸检测运行脚本

echo "=========================================="
echo "YOLOv9 批量人脸检测脚本"
echo "=========================================="

# 检查Python环境
echo "检查Python环境..."
python3 --version
if [ $? -ne 0 ]; then
    echo "错误: Python3 未安装或不在PATH中"
    exit 1
fi

# 检查必要文件
echo "检查必要文件..."
if [ ! -f "batch_face_detection.py" ]; then
    echo "错误: batch_face_detection.py 文件不存在"
    exit 1
fi

if [ ! -f "yolov9/yolov9.pt" ]; then
    echo "错误: yolov9/yolov9.pt 模型文件不存在"
    exit 1
fi

# 检查数据集目录
echo "检查数据集目录..."
if [ ! -d "/data/txf/1/data" ]; then
    echo "错误: 数据集目录 /data/txf/1/data 不存在"
    exit 1
fi

# 修复PyTorch 2.6兼容性问题
echo "检查并修复PyTorch兼容性问题..."
python3 fix_pytorch_compatibility.py

# 运行测试
echo "运行功能测试..."
python3 test_face_detection.py

# 询问是否继续
echo ""
read -p "是否继续运行批量人脸检测？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消运行"
    exit 0
fi

# 运行批量检测
echo "开始批量人脸检测..."
python3 batch_face_detection.py \
    --data-dir /data/txf/1/data \
    --weights /data/txf/1/yolov9-face-detection-main/yolov9/yolov9.pt \
    --conf-threshold 0.25

echo "批量人脸检测完成！" 