import os
import shutil

# 指定源目录和目标目录
src_dir = r"D:\Code\Python\YOLOv8-Tou\data\0309"
dst_dir = r"D:\Code\Python\YOLOv8-Tou\data\images"

# 确保目标目录存在,如果不存在则创建
os.makedirs(dst_dir, exist_ok=True)

# 遍历源目录
for root, dirs, files in os.walk(src_dir):
    for file in files:
        # 检查文件扩展名是否为.json
        if file.endswith(".png"):
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_dir, file)
            # 移动文件
            shutil.copy(src_file, dst_file)
            print(f"Moved {file} to {dst_dir}")