import sys
import numpy as np

assert len(sys.argv)==2

with open(sys.argv[1], "r") as ifs:
    lines = ifs.readlines()

all_data = []
index_arr = []
for idx, line in enumerate(lines):
    if line.startswith("Round"):
        all_data.append([[], []]) # Total_time, recall
        method = lines[idx+1].split()[0]
        index_arr.append(method)
    elif len(line.split()) == 4:
        content = line.split()
        all_data[-1][0].append(float(content[3]))
        all_data[-1][1].append(float(content[1]))

with open("result.txt", "w") as ofs:
    for idx, item in enumerate(all_data):
        if len(item[0])==10:
            ofs.write(f"{index_arr[idx]}-time\t")
            for i in item[0]:
                ofs.write(f"{i}\t")
            ofs.write("\n")
            ofs.write(f"{index_arr[idx]}-recall\t")
            for i in item[1]:
                ofs.write(f"{i}\t")
            ofs.write("\n")