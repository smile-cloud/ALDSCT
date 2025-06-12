# def sort_nested_list(lst):
#     return sorted(lst, key=lambda x: x[0])
#
#
# a = [[1, 2, 3], [0.5, 0.3, 0.2]]
# print(sort_nested_list(a))  # 输出：[[0.5, 0.3, 0.2], [1, 2, 3]]


# # 定义范围
# ranges = [(11, 18), (21, 28), (31, 38), (41, 48)]
#
# # 设置左侧列的计数器从0到31
# left_column = 0
#
# # 打印每个范围中的数字
# for start, end in ranges:
#     for i in range(start, end + 1):
#         print(f"{left_column}: {i}")
#         left_column += 1


# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 自定义类别标签
# # labels = [
# #     "nasal cavity",
# #     "nasopharynx",
# #     "adenoid",
# #     "epiglottic pharyngeal fold",
# #     "whole laryngopharynx",
# #     "pyriform fossa",
# #     "vocal fold"
# # ]
#
# labels = [
#     "hypopharynx",
#     "nasal cavity",
#     "nasopharynx"
# ]
#
#
# # 自定义对角线数据
# diagonal_values = [209+292+292+291+291+291+291+291+291+291,
#                    205+210+224+235+234+235+238+236+237+237,
#                    221+222+229+233+234+236+236+236+237+237
#                    ]
#
# # 构造矩阵，并用 NaN 填充非对角线部分（让它们不显示）
# cm = np.full((len(labels), len(labels)), np.nan)
# for i in range(len(labels)):
#     cm[i, i] = diagonal_values[i]
#
# # 画图
# plt.figure(figsize=(10, 8))  # 调整宽高，避免太大或太小
# ax = sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', xticklabels=labels, yticklabels=labels, vmin=0, vmax=3000)
#
# # 设置轴标签
# plt.xlabel("True Labels", fontsize=12)
# plt.ylabel("Predicted Labels", fontsize=12)
# plt.title("Confusion Matrix", fontsize=16)
#
# # 旋转 x 轴标签以防止重叠
# plt.xticks(rotation=45, ha="right", fontsize=12)
# plt.yticks(rotation=0, fontsize=12)
#
# # 调整边距，防止标签被截断
# plt.subplots_adjust(left=0.3, bottom=0.3, right=1, top=0.9)
#
# plt.show()

import os
import json
import glob

# JSON 文件所在的文件夹路径（请修改为你的路径）
json_folder = r"D:\Code\Python\YOLOv8-Tou\data\json"

# 用于存储所有标签
all_labels = set()

# 遍历文件夹中的所有 JSON 文件
for json_file in glob.glob(os.path.join(json_folder, "*.json")):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取所有 `label` 并加入集合（去重）
    for shape in data.get("shapes", []):
        if "label" in shape:
            all_labels.add(shape["label"])

# 转换为列表
label_list = sorted(list(all_labels))  # 排序方便查看

# 输出标签列表
print("所有标签:", label_list)


