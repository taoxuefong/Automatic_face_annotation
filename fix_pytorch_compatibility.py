#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复PyTorch 2.6兼容性问题的脚本
"""

import os
import re
import sys
from pathlib import Path

def fix_torch_load_calls():
    """修复torch.load调用，添加weights_only=False参数"""
    
    # 需要修复的文件列表
    files_to_fix = [
        'yolov9/models/experimental.py',
        'yolov9/hubconf.py'
    ]
    
    print("修复PyTorch 2.6兼容性问题...")
    
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过")
            continue
            
        print(f"处理文件: {file_path}")
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修复torch.load调用
        # 匹配 torch.load(...) 但不包含 weights_only 参数的调用
        pattern = r'torch\.load\(([^)]+), map_location=([^)]+)\)'
        replacement = r'torch.load(\1, map_location=\2, weights_only=False)'
        
        new_content = re.sub(pattern, replacement, content)
        
        # 如果内容有变化，写回文件
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  ✓ 已修复 {file_path}")
        else:
            print(f"  - 无需修复 {file_path}")
    
    print("PyTorch 2.6兼容性修复完成！")

def check_pytorch_version():
    """检查PyTorch版本"""
    try:
        import torch
        version = torch.__version__
        print(f"当前PyTorch版本: {version}")
        
        # 检查是否是2.6或更高版本
        major, minor = map(int, version.split('.')[:2])
        if major >= 2 and minor >= 6:
            print("检测到PyTorch 2.6+，需要应用兼容性修复")
            return True
        else:
            print("PyTorch版本低于2.6，无需修复")
            return False
    except ImportError:
        print("警告: 无法导入torch模块")
        return False

def main():
    print("PyTorch 2.6兼容性修复工具")
    print("=" * 40)
    
    # 检查PyTorch版本
    if not check_pytorch_version():
        print("无需修复，退出")
        return
    
    # 修复torch.load调用
    fix_torch_load_calls()
    
    print("\n修复完成！现在可以运行批量人脸检测脚本了。")
    print("运行命令:")
    print("python3 batch_face_detection.py")

if __name__ == "__main__":
    main() 