#!/bin/bash

# YOLOv8批量人体检测运行脚本

echo "=========================================="
echo "YOLOv8 批量人体检测脚本"
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
if [ ! -f "batch_person_detection.py" ]; then
    echo "错误: batch_person_detection.py 文件不存在"
    exit 1
fi

if [ ! -f "yolov8-detect/best.pt" ]; then
    echo "错误: yolov8-detect/best.pt 模型文件不存在"
    exit 1
fi

# 检查数据集目录
echo "检查数据集目录..."
if [ ! -d "/data/txf/1/data" ]; then
    echo "错误: 数据集目录 /data/txf/1/data 不存在"
    exit 1
fi

# 运行测试
echo "运行功能测试..."
python3 test_person_detection.py

# 询问是否继续
echo ""
read -p "是否继续运行批量人体检测？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消运行"
    exit 0
fi

# 运行批量检测
echo "开始批量人体检测..."
python3 batch_person_detection.py \
    --data-dir /data/txf/1/data \
    --weights yolov8-detect/best.pt \
    --conf-threshold 0.5

echo "批量人体检测完成！" 