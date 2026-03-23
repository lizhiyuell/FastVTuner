import sys
import os
import matplotlib.pyplot as plt

# get the Pareto results from array
def get_Pareto_result_from_array(tput_arr, recall_arr):
    res = []
    for tput, recall in zip(tput_arr, recall_arr):
        res.append([tput, recall])
    res = sorted(res, key=lambda x : x[0])

    tput_arr = []
    recall_arr = []

    pos = len(res)-1
    max_recall = 0
    while pos >= 0:
        if res[pos][1] > max_recall:
            max_recall = res[pos][1]
            tput_arr.append(res[pos][0])
            recall_arr.append(res[pos][1])
        pos -= 1

    return tput_arr, recall_arr

for f in os.listdir("./"):
    if f.endswith(".log"):
        res_path = f.replace(".log", ".jpg")
        plt.clf()
        with open(f, "r") as ifs:
            lines = ifs.readlines()
            for i in range(0, len(lines), 2):
                content = lines[i].split()
                content2 = lines[i+1].split()
                key = content[0]
                percent = key.split("-")[0]

                tput = [float(item) for item in content[1:] if item!="0"]
                recall = [float(item) for item in content2[1:] if item!="0"]

                tput, recall = get_Pareto_result_from_array(tput, recall)

                if percent=="p100":
                    plt.plot(tput, recall, label=percent)
                else:
                    plt.scatter(tput, recall, label=percent)
            
            plt.xlabel("Throughput")
            plt.ylabel("Recall")
            plt.legend()

            plt.savefig(res_path)
