import  sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

assert len(sys.argv)==2

res_small_recall = defaultdict(list)
res_large_recall = defaultdict(list)
with  open(sys.argv[1], "r") as ifs:
    ifs.readline()
    for line in ifs:
        # time
        content_time = line.split()
        key_time = content_time[0]+content_time[1]
        val_time = [float(x) for x in content_time[2:]]
        line = ifs.readline()
        content_recall = line.split()
        key_recall = content_recall[0]+content_recall[1]
        val_recall = [float(x) for x in content_recall[2:]]
        assert key_time.split("-")[1] == key_recall.split("-")[1]

        if val_recall[-1] < 0.7:
            res_small_recall[key_time].append(val_time)
            res_small_recall[key_recall].append(val_recall)
        else:
            res_large_recall[key_time].append(val_time)
            res_large_recall[key_recall].append(val_recall)


def get_method(string):
    start_pos  = string.find("'") + 1
    return string[start_pos:-2]

color_map = {
    "IVF_PQ" : "b",
    "IVF_SQ8" : "k",
    "IVF_FLAT" : "g",
    "SCANN" : "r"
}

# 我们按照不同的取值范围，分别来看: 大于等于50%召回率和小于50%召回率

res_arr = [res_small_recall, res_large_recall]
name_arr = ["low_recall", "large_recall"]

for i in range(len(res_arr)):
    all_res = res_arr[i]
    name = name_arr[i]
    # plot the result
    plt.clf()
    for key, value in all_res.items():
        if "Time" in key:
            for item in value:
                if len(item) != 10:
                    continue
                x = [x * 10 for x in range(len(item))]
                y = np.array(item) /  item[-1]
                plt.plot(x, y, color=color_map[get_method(key)], linewidth=1)
    plt.xlabel("Dataset  Percent")
    plt.ylabel("Relative to full dataset")
    plt.savefig(f"plot_time_{name}.png")

    plt.clf()
    for key, value in all_res.items():
        if "Recall" in key:
            for  item in value:
                if len(item) != 10:
                    continue
                x = [x * 10 for x in range(len(item))]
                y = np.array(item) /  item[-1]
                plt.plot(x, y, color=color_map[get_method(key)], linewidth=1)
    plt.xlabel("Dataset  Percent")
    plt.ylabel("Relative to full dataset")
    plt.savefig(f"plot_recall_{name}.png")