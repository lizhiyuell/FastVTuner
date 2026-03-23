import sys
import numpy as np
import matplotlib.pyplot as plt

assert len(sys.argv)==2
build_time = []
search_time = []
other_time = []
with open(sys.argv[1], "r") as ifs:
    for line in ifs:
        content = line.split()
        build_time.append(float(content[-5]))
        search_time.append(float(content[-4]))
        other_time.append(float(content[-1]) - build_time[-1] - search_time[-1])
        assert other_time[-1] > 0

print(sum(build_time))
print(sum(search_time))
print(sum(other_time))
print(f"总时间：{(sum(build_time) + sum(search_time) + sum(other_time)) / 3600} h")

x = np.arange(len(build_time))
plt.figure(figsize=(16, 6))
plt.bar(x, build_time, label='build_time')
plt.bar(x, search_time, bottom=build_time, label='search_time')
plt.bar(x, other_time, bottom=np.array(build_time)+np.array(search_time), label='other_time')

plt.legend()
plt.xlabel("Round")
plt.ylabel("Time/s")

plt.savefig("analyze.png")