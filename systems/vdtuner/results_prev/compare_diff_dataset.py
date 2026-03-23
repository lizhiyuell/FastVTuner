# 通过绘制帕累托前沿的tput-recall曲线，比较不同数据集规模下的性能好坏

import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

assert len(sys.argv)==2


def get_Pareto_result(filename):
    # [tput, recall]
    res = []

    # 首先采集所有值
    with open(filename, "r") as ifs:
        for line in ifs:
            content = line.split()
            tput = float(content[-3])
            recall = float(content[-2])
            res.append([tput, recall])

    # 然后，去掉非帕累托前沿值，首先按照某一维度（如tput）从小到大
    res = sorted(res, key=lambda x : x[0])

    # 最后，从后往前扫描，将回撤点删除即可，并改成tput和recall分别存储的方式
    final = [[], []]
    max_recall = 0
    pos = len(res)-1
    while pos >= 0:
        t, r = res[pos][0], res[pos][1]
        if r > max_recall:
            final[0].append(t)
            final[1].append(r)
        max_recall = max(max_recall, r)
        pos -= 1

    final[0].reverse()
    final[1].reverse()

    return final

all_res = defaultdict(list)
for f in os.listdir(sys.argv[1]):
    if not f.endswith(".log") or not f.startswith("record"):
        continue
    
    dataset_name = f.split("_")[1]

    # 提取出原始数据集的信息和数据集大小
    pos = dataset_name.find("-p-")
    if pos==-1:
        percent = 100
    else:
        percent = int(dataset_name[pos+3:])
        dataset_name = dataset_name[:pos]
    
    # 读取文件并获得tput-recall的帕累托前沿
    res = get_Pareto_result(os.path.join(sys.argv[1], f))

    all_res[dataset_name].append([res, percent])


output_dir = "plots"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
# 绘制结果分析图像
for key, value in all_res.items():
    plt.clf()
    for item in value:
        plt.plot(item[0][0], item[0][1], label=f"{item[1]}% dataset")
    
    plt.legend()
    plt.xlabel("Throughput/QPS")
    plt.ylabel("Recall/%")

    plt.savefig(os.path.join(output_dir, f"{key}.png"))