import os
import time
from ultralytics import YOLO
import cv2
import math
import numpy as np

# 配置参数
model_path = r"D:\Code\Python\YOLOv8-Tou\runs\pose\train\weights\best.pt"  # 你的YOLOv8模型路径
input_folder = r"D:\Code\Python\YOLOv8-Tou\data\test"  # 需要预测的图片文件夹
output_folder = "results"  # 预测结果保存文件夹

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

def angle_between_three_points(A, B, C):
    """
    计算点 B 处的夹角 ∠ABC（单位为度）
    A, B, C 是二维或三维空间中的点，格式为 (x, y) 或 (x, y, z)
    """
    # 构造向量 BA 和 BC
    BA = [a - b for a, b in zip(A, B)]
    BC = [c - b for c, b in zip(C, B)]

    # 计算点积
    dot_product = sum(a * b for a, b in zip(BA, BC))

    # 计算向量的模长
    magnitude_BA = math.sqrt(sum(a ** 2 for a in BA))
    magnitude_BC = math.sqrt(sum(b ** 2 for b in BC))

    # 防止除以零
    if magnitude_BA == 0 or magnitude_BC == 0:
        raise ValueError("某个向量的长度为0，无法计算夹角")

    # 计算夹角的余弦值，并限制在 [-1, 1] 之间避免浮点误差
    cos_angle = max(min(dot_product / (magnitude_BA * magnitude_BC), 1), -1)

    # 计算角度（弧度转度）
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def angle_between_lines(pt1_start, pt1_end, pt2_start, pt2_end):
    """
    计算两条线段之间的夹角（单位：度）
    """
    vec1 = np.array(pt1_end) - np.array(pt1_start)
    vec2 = np.array(pt2_end) - np.array(pt2_start)

    # 点积
    dot = np.dot(vec1, vec2)
    # 模长乘积
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    # 防止数值精度问题导致 acos 域错误
    cos_theta = np.clip(dot / norm_product, -1.0, 1.0)

    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def project_point_to_line(point, line_pt1, line_pt2):
    """
    将一个点投影到一条线段上，返回垂足坐标
    """
    point = np.array(point)
    A = np.array(line_pt1)
    B = np.array(line_pt2)
    AB = B - A
    AP = point - A
    t = np.dot(AP, AB) / np.dot(AB, AB)
    projection = A + t * AB
    return projection

def signed_distance_to_perpendicular(A, N, P, Or):
    """
    输入：
        A: 要测距的点
        N: 鼻根点
        P, Or: 定义FH平面的两点
    输出：
        A 到 NP 垂线的有符号距离
    """

    # 构造 FH 平面的方向向量（从 Or 指向 P）
    FH_dir = np.array(P) - np.array(Or)

    # 单位化
    FH_unit = FH_dir / np.linalg.norm(FH_dir)

    # 计算 NP 的方向（垂直于 FH 平面，通过 N 点）
    # 即 FH 的法向量旋转 90 度（逆时针）
    NP_dir = np.array([-FH_unit[1], FH_unit[0]])

    # 构造 A - N 的向量
    AN = np.array(A) - np.array(N)

    # 投影 A-N 向量到 NP_dir（即正交方向）
    # 投影大小就是有符号距离
    signed_dist = np.dot(AN, NP_dir)

    return signed_dist


def point_to_line_distance(point, line_start, line_end):
    """
    计算一个点到一条直线（由 line_start 和 line_end 定义）的垂直距离
    """
    P = np.array(point)
    A = np.array(line_start)
    B = np.array(line_end)

    AB = B - A
    AP = P - A

    # 叉积的模（2D）
    cross = abs(AB[0] * AP[1] - AB[1] * AP[0])
    distance = cross / np.linalg.norm(AB)

    return distance

label_mapping = {
    0: 'A', 1: 'B', 2: 'Cm', 3: 'Gn', 4: 'LL', 5: 'Me', 6: 'N', 7: 'Or', 8: 'P',
    9: 'Pog', 10: "Pog'", 11: 'Prn', 12: 'S', 13: 'Si', 14: 'Sn', 15: 'UL', 16: 'r1', 17: 'r2', 18: 'OP-1', 19: 'OP-2',
    20: 'MP-1', 21: 'MP-2',
}
# print(label_mapping.values())
# ✅ 定义关键点颜色
keypoint_colors = {
    "A": (255, 0, 0), "B": (0, 255, 0), "Cm": (0, 0, 255), "Gn": (255, 255, 0),
    "LL": (255, 0, 255), "Me": (0, 255, 255), "N": (128, 0, 0), "Or": (128, 128, 0),
    "P": (128, 0, 128), "Pog": (0, 128, 128), "Pog'": (200, 100, 50), "Prn": (50, 100, 200),
    "S": (100, 50, 200), "Si": (200, 50, 100), "Sn": (100, 200, 50), "UL": (50, 200, 100),
    "r1": (0, 50, 200), "r2": (0, 50, 200), "OP_1": (0, 128, 0), "OP_2": (0, 0, 128),
    "MP_1": (0, 128, 0), "MP_2": (0, 0, 128)
}

# 加载模型
model = YOLO(model_path)

# 遍历输入文件夹中的所有图片
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    output_path = os.path.join(output_folder, image_name)

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取文件: {image_name}")
        continue

    # 记录开始时间
    start_time = time.time()

    # 进行预测
    results = model(image)

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{image_name} 预测时间: {elapsed_time:.4f} 秒")

    # ✅ 获取预测结果
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()  # 获取关键点坐标 (x, y)
        # print(keypoints)
        points = np.array(keypoints)
        labels = ['A', 'B', 'Cm', 'Gn', 'LL', 'Me', 'N', 'Or', 'P', 'Pog', "Pog'", 'Prn',
                  'S', 'Si', 'Sn', 'UL', 'r1', 'r2', 'OP-1', 'OP-2', 'MP-1', 'MP-2']

        A = points[0][0]
        B = points[0][1]
        Cm = points[0][2]
        Gn = points[0][3]
        LL = points[0][4]
        Me = points[0][5]
        N = points[0][6]
        Or = points[0][7]
        P = points[0][8]
        Pog = points[0][9]
        Pog_ = points[0][10]  # 替代了 "Pog'"，不能直接作为变量名
        Prn = points[0][11]
        S = points[0][12]
        Si = points[0][13]
        Sn = points[0][14]
        UL = points[0][15]
        r1 = points[0][16]
        r2 = points[0][17]
        OP_1 = points[0][18]
        OP_2 = points[0][19]
        MP_1 = points[0][20]
        MP_2 = points[0][21]

        SNA = angle_between_three_points(S, N, A)
        SNB = angle_between_three_points(S, N, B)
        ANB = angle_between_three_points(A, N, B)
        CmSnUL = angle_between_three_points(Cm, Sn, UL)

        # 计算垂足
        AO = project_point_to_line(A, OP_1, OP_2)
        BO = project_point_to_line(B, OP_1, OP_2)
        # 计算垂足之间的距离
        AO_BO = np.linalg.norm(AO - BO)

        A_Np = signed_distance_to_perpendicular(A, N, P, Or)
        Po_Np = signed_distance_to_perpendicular(Pog, N, P, Or)
        SGn_FH = angle_between_lines(S, Gn, Or, P)
        Si_H = point_to_line_distance(Si, UL, Pog_)
        Pog__TVL = Sn[0] - Pog_[0]
        UL_EP = point_to_line_distance(UL, Prn, Pog_)
        LL_EP = point_to_line_distance(LL, Prn, Pog_)

        results = {
            'SNA': SNA,
            'SNB': SNB,
            'ANB': ANB,
            'CmSnUL': CmSnUL,
            'AO_BO': AO_BO,
            'A_Np': A_Np,
            'Po_Np': Po_Np,
            'SGn_FH': SGn_FH,
            'Si_H': Si_H,
            'Pog__TVL': Pog__TVL,
            'UL_EP': UL_EP,
            'LL_EP': LL_EP,
        }

        # 写入到 txt 文件
        with open(output_folder + "/" + image_name + ".txt", "w") as f:
            for name, value in results.items():
                f.write(f"{name}: {value:.4f}\n")

        for keypoint_set in keypoints:  # 遍历多个检测目标
            for idx, (x, y) in enumerate(keypoint_set):
                if x == 0 and y == 0:  # 跳过不可见点
                    continue

                # 获取颜色
                color = keypoint_colors.get(list(keypoint_colors.keys())[idx], (0, 255, 255))
                print(list(keypoint_colors.keys())[idx], ': ', color)

                # 绘制关键点
                cv2.circle(image, (int(x), int(y)), 10, color, -1)

                # 显示关键点名称
                cv2.putText(image, list(keypoint_colors.keys())[idx], (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

    # 保存预测结果图片
    cv2.imwrite(output_path, image)

print("预测完成，所有结果已保存。")