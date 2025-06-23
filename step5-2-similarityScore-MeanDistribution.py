import pickle

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm

def similarityCompute(filename):
    loaded_similarity_matrix = np.load(filename)

    # 对每一行的元素进行排序（从高到低）
    sorted_indices = np.argsort(loaded_similarity_matrix, axis=1)[:, ::-1]
    sorted_similarity_matrix = np.take_along_axis(loaded_similarity_matrix, sorted_indices, axis=1)
    num_lists = sorted_similarity_matrix.shape[1]  # 获取列数

    # 初始化存储每个 k 的平均值 A 和总体平均值 B
    k_values = np.arange(1, num_lists + 1)  # 将 k_values 转换为 ndarray

    average_B = []

    print("计算过程：")
    for k in k_values:
        # 计算每一行前 k 个最大值的平均值 A
        top_k_averages = np.mean(sorted_similarity_matrix[:, :k], axis=1)
        # 计算这些平均值 A 的总体平均值 B
        overall_average_B = np.mean(top_k_averages)
        average_B.append(overall_average_B)
        if k % 500 == 0:
            print(f"已完成 k={k}")

    # 将 average_B 转换为 ndarray
    average_B = np.array(average_B)

    # 动态加载系统中的字体
    font = fm.FontProperties(fname=fm.findfont('SimHei'))  # 或者使用 'Microsoft YaHei'

    # 绘图代码
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, average_B, marker='o', markersize=2, linestyle='-', color='b', label="总体平均值 B")
    plt.title("总体平均值 B 随选取的元素数量 k 的变化", fontproperties=font)
    plt.xlabel("选取的元素数量 k", fontproperties=font)
    plt.ylabel("总体平均值 B", fontproperties=font)
    interval = 200
    plt.xticks(k_values[::interval], rotation=45)
    plt.grid(True)
    plt.legend(prop=font)
    plt.show()


if __name__ == '__main__':

    filename = "E:/LLM-Project/算法设计/IFT-Select-necessarily（2025-3-13）/Singal pass-based IFT/train/data/MultiTaskCorrelation-checkpoint/train_similarity_matrix.npy"
    similarityCompute(filename)