import cv2
import os
import glob
from PIL import Image
import numpy as np

# 设置路径
image_dir = r"D:\Code\Python\YOLOv8-Tou\data\images"  # 原始图片路径
label_dir = r"D:\Code\Python\YOLOv8-Tou\data\labels"  # YOLO 标签路径
output_dir = r"D:\Code\Python\YOLOv8-Tou\data\visualized"  # 输出带有关键点的图片
os.makedirs(output_dir, exist_ok=True)

label_mapping = {
    0: 'A', 1: 'B', 2: 'Cm', 3: 'Gn', 4: 'LL', 5: 'Me', 6: 'N', 7: 'Or', 8: 'P',
    9: 'Pog', 10: "Pog'", 11: 'Prn', 12: 'S', 13: 'Si', 14: 'Sn', 15: 'UL', 16: 'r1', 17: 'r2', 18: 'OP-1', 19: 'OP-2',
    20: 'MP-1', 21: 'MP-2',
}

# 处理所有图片
for txt_file in glob.glob(os.path.join(label_dir, "*.txt")):
    image_name = os.path.basename(txt_file).replace(".txt", ".jpg")  # 假设图片格式为 .jpg
    image_path = os.path.join(image_dir, image_name)

    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"⚠️ 图片缺失: {image_path}")
        continue

    # 读取图片
    img = cv2.imread(image_path)

    with Image.open(image_path) as img:
        width, height = img.size  # 获取宽度和高度
        img = np.array(img)  # 转换为 NumPy 数组
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV 使用 BGR 颜色空间

    # 读取关键点信息
    with open(txt_file, "r") as f:
        line = f.readline().strip().split()  # 只读取第一行
        keypoints = line[5:]  # 前 5 个是目标框信息，后面的是关键点

    # 解析关键点坐标
    for i in range(0, len(keypoints), 3):
        x, y, v = float(keypoints[i]), float(keypoints[i + 1]), int(keypoints[i + 2])
        if v == 2:  # 仅可见的点进行绘制
            px, py = int(x * width), int(y * height)
            cv2.circle(img, (px, py), 8, (255, 50, 100), -1)  # 绿色关键点
            cv2.putText(img, str(label_mapping[i // 3]), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 100, 255), 1)

    # 保存可视化结果
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, img)
    print(f"✅ 关键点可视化完成: {output_path}")
