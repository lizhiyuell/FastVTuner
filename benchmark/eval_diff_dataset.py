# 测试随机选定的参数下，不同数据集规模对结果的影响

# simple_random_tester.py
import sys
sys.path.append("..")

import os
import json
import random
import subprocess as sp
import numpy as np

from utils import RealEnv, KNOB_PATH, RUN_ENGINE_PATH

# 与 main_tuner.py 中保持一致
INDEX_PARAM_PATH = '/root/lzy/FVDB_tuning/VDTuner/auto-configure/index_param.json'

# 根据向量维度更新 m 的可选取值（与 main_tuner.py 相同逻辑）
def update_m_with_dimension(dimension):
    # ① 求所有因数，并排序
    factors = []
    for i in range(1, dimension + 1):
        if dimension % i == 0:
            factors.append(i)
    factors.sort()

    # ② 选出最接近 10 的那个作为 default
    default_value = min(factors, key=lambda x: abs(x - 10))

    # ③ 读取 index_param.json 和 whole_param.json，更新 m.default 和 m.enum_values
    for filename in [INDEX_PARAM_PATH, KNOB_PATH]:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["m"]["default"] = default_value
        data["m"]["enum_values"] = factors

        # ④ 覆盖写回原文件
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():

    # total_rounds = 50
    total_rounds = 20
    result_dir = "./diff_size_result"
    # result_dir = "./temp_result"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 原始数据集配置：名称、dimension
    run_params_list = [
        # ["deep-image-96-angular", 96],
        ["glove-100-angular", 100]
    ]

    # 子集百分比：-p-10, -p-20, ..., -p-100
    percents = list(range(10, 101, 10))
    # percents = [10, 100]

    for dataset_name, dimension in run_params_list:
        print(f"\n=== 基础数据集: {dataset_name}, dim={dimension} ===")

        # 清空历史log信息
        record_file_name = os.path.join(result_dir, f"record_{dataset_name}.csv")
        log_file = os.path.join(result_dir, f"record_{dataset_name}.txt")
        if os.path.isfile(record_file_name):
            os.remove(record_file_name)
        if os.path.isfile(log_file):
            os.remove(log_file)

        ofs_result = open(record_file_name, "w")
        ofs = open(log_file, "w")

        # 根据维度更新 m 的取值
        update_m_with_dimension(dimension)

        # 构造一组随机参数
        env = RealEnv()
        knob_names = env.names
        knob_num = len(knob_names)

        # 一开始为结果文件写入表头
        for knob_name in knob_names:
            ofs.write(f"{knob_name}\t")
        ofs.write("\n")

        ofs_result.write("Dataset_percentage")
        for pp in percents:
            ofs_result.write(f"\t{pp}%")
        ofs_result.write("\n")

        # 只做聚类索引
        target_index = ["IVF_FLAT", "IVF_SQ8", "IVF_PQ", "SCANN"]
        # target_index = ["IVF_FLAT"]

        # 执行多轮参数调节
        for r in range(1, total_rounds + 1):
            ofs.write(f"Round {r}/{total_rounds}\n")

            # 限定索引类型
            while 1:
                # 在 [0,1] 上为每个 knob 随机取一个值
                x = [random.random() for _ in range(knob_num)]
                idx_name = env.knob_stand.scale_back(knob_names[0], x[0])
                if idx_name[1] in target_index:
                    break

            # 我们对nlist和nprobe进行修正，保证大小关系
            nlist = env.knob_stand.scale_back(knob_names[1], x[1])[1]
            nprobe = env.knob_stand.scale_back(knob_names[2], x[2])[1]
            nprobe = max(1, nprobe)
            if nprobe > nlist:
                nprobe = nlist
            x[1] = env.knob_stand.scale_forward(knob_names[1], nlist)
            x[2] = env.knob_stand.scale_forward(knob_names[2], nprobe)

            all_real_vals = []
            # 计算所有 knob 的真实取值（用于写入结果文件）
            for j, knob_name in enumerate(knob_names):
                _, real_val = env.knob_stand.scale_back(knob_name, x[j])
                all_real_vals.append(real_val)
                # 将参数写入文件
                ofs.write(f"{real_val}\t")
            ofs.write("\n")

            # 将参数配置到系统中
            env.config_system_with_params(x)
            original_nlist = all_real_vals[1]

            # 针对一个配置，运行三次脚本
            # 最终结果
            test_search_time = np.zeros(len(percents))
            test_search_recall = np.zeros(len(percents))
            # 是否出错而需要终止这组结果
            should_stop = False
            for times in range(3):
            # for times in range(1):
                # 对于每一个测试子集分析
                for idx_p, percent in enumerate(percents):
                    subset_name = f"{dataset_name}-p-{percent}"
                    ofs.write(f"##Data_percent: {percent} %\n")

                    # 对于IVF系列的索引，我们会保证每一个partition种索引参数是不变的。我们假设生成的这个nlist是给100p的，所以我们当前的nlist也对应发生变化
                    print(all_real_vals[0])
                    if all_real_vals[0] in ["IVF_FLAT", "IVF_SQ8", "IVF_PQ", "SCANN"]:
                        new_n_list = max(1, round(original_nlist * percent / 100))
                        ofs.write(f"[Reconf nlist] {original_nlist} -> {new_n_list}\n")
                        x[1] = env.knob_stand.scale_forward(knob_names[1], new_n_list)
                        env.config_system_with_params(x)

                    subdataset_name = f"{dataset_name}-p-{percent}"
                    run_cmd = f'sudo timeout 1800 {RUN_ENGINE_PATH} "" "" {subdataset_name}'

                    try:
                        result = sp.run(run_cmd, shell=True, stdout=sp.PIPE)
                        result = result.stdout.decode().split()

                        # Get the result
                        tput, recall, load_time, search_time = float(result[-2]), float(result[-3]), float(result[1]), float(result[2])
                    except:
                        tput, recall, load_time, search_time = 0, 0, 0, 0
                        # 目前的逻辑：一个参数出了问题，所有参数都跳过，进入下一轮
                        ofs.write(f"Detect failure and goona skip all the tests of this configuration\n")
                        should_stop = True
                        break

                    # Output the result
                    ofs.write(f"{tput} {recall} {load_time} {search_time}\n")
                    ofs.flush()
                    test_search_time[idx_p] += search_time
                    test_search_recall[idx_p] += recall
                
                if should_stop:
                    break
            
            # 对于完整的测试，保存结果
            if not should_stop:
                ofs_result.write(f"Time-{idx_name[1]}")
                for item in test_search_time:
                    ofs_result.write(f"\t{item/3}")
                ofs_result.write("\n")
                ofs_result.write(f"Recall-{idx_name[1]}")
                for item in test_search_recall:
                    ofs_result.write(f"\t{item/3}")
                ofs_result.write("\n")
                ofs_result.flush()

        ofs.close()
        ofs_result.close()

if __name__ == "__main__":
    main()