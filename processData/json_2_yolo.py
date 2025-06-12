import os
import json
import glob

# 关键点类别映射
label_mapping = {
    'A': 0, 'B': 1, 'Cm': 2, 'Gn': 3, 'LL': 4, 'Me': 5, 'N': 6, 'Or': 7, 'P': 8,
    'Pog': 9, "Pog'": 10, 'Prn': 11, 'S': 12, 'Si': 13, 'Sn': 14, 'UL': 15, 'r1': 16, 'r2': 17, 'OP': 18, 'MP': 20,
}

NUM_KEYPOINTS = 22  # OP\MP 有两个点，因此总关键点数为 22

# JSON 文件夹路径
input_json_dir = r"D:\Code\Python\YOLOv8-Tou\data\json"  # 你的 JSON 目录
output_txt_dir = r"D:\Code\Python\YOLOv8-Tou\data\labels"  # 输出 YOLO 标签目录
os.makedirs(output_txt_dir, exist_ok=True)

# 处理所有 JSON 文件
for json_file in glob.glob(os.path.join(input_json_dir, "*.json")):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    objects = [[0, 0, 0] for _ in range(NUM_KEYPOINTS)]  # 初始化 20 个关键点
    IMG_WIDTH = data["imageWidth"]
    IMG_HEIGHT = data["imageHeight"]

    keypoints = {}

    # 遍历 shapes
    for shape in data.get("shapes", []):
        label = shape["label"]
        if label not in label_mapping:
            continue

        points = shape["points"]
        idx = label_mapping[label]

        if label == "OP" and len(points) == 2:
            keypoints[label] = [(p[0] / IMG_WIDTH, p[1] / IMG_HEIGHT) for p in points]
            objects[idx] = [keypoints[label][0][0], keypoints[label][0][1], 2]
            objects[idx + 1] = [keypoints[label][1][0], keypoints[label][1][1], 2]
        elif label == "MP" and len(points) == 2:
            keypoints[label] = [(p[0] / IMG_WIDTH, p[1] / IMG_HEIGHT) for p in points]
            objects[idx] = [keypoints[label][0][0], keypoints[label][0][1], 2]
            objects[idx + 1] = [keypoints[label][1][0], keypoints[label][1][1], 2]
        else:
            x, y = points[0]
            keypoints[label] = (x / IMG_WIDTH, y / IMG_HEIGHT)
            objects[idx] = [keypoints[label][0], keypoints[label][1], 2]

    # 获取 P, Prn, Me, N 关键点的坐标，并归一化
    if "P" in keypoints and "Prn" in keypoints and "Me" in keypoints and "N" in keypoints:
        x_min = min(keypoints["P"][0], keypoints["Prn"][0])
        x_max = max(keypoints["P"][0], keypoints["Prn"][0])
        y_min = min(keypoints["Me"][1], keypoints["N"][1])
        y_max = max(keypoints["Me"][1], keypoints["N"][1])

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
    else:
        x_center, y_center, width, height = 0.5, 0.5, 1.0, 1.0  # 关键点缺失时默认值

    class_id = 0  # 所有目标使用 class_id = 0
    yolo_format = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    for obj in objects:
        yolo_format += f" {obj[0]:.6f} {obj[1]:.6f} {obj[2]}"

    # 保存 YOLO 格式 .txt
    save_path = os.path.join(output_txt_dir, os.path.basename(json_file).replace(".json", ".txt"))
    with open(save_path, "w") as f:
        f.write(yolo_format)

    print(f"✅ 转换完成: {save_path}")
