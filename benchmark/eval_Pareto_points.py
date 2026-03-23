# read the results on sampled datasets, and using the Pareto configuration to re-benchmark the full-dataset

from utils import *
import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

assert len(sys.argv)==2

# save the mapping between dataset and dimension
dim_map = {
    "deep-image-96-angular" : 96,
    "gist-960-euclidean" : 960,
    "glove-100-angular" : 100
}

# 1. parsing the result files and generate the points to test
all_res = defaultdict(list)
for f in os.listdir(sys.argv[1]):
    if not f.endswith(".log") or not f.startswith("record"):
        continue
    
    dataset_name = f.split("_")[1]

    pos = dataset_name.find("-p-")
    if pos==-1:
        percent = 100
    else:
        percent = int(dataset_name[pos+3:])
        dataset_name = dataset_name[:pos]

    # maybe we should reconfigure the M according to dataset information
    dim = dim_map[dataset_name]
    update_m_with_dimension(dim)

    res = get_Pareto_result(os.path.join(sys.argv[1], f))

    all_res[dataset_name].append([res, percent])


# 2. for dataset less than the full size, we re-evaluate its result
output_dir = "result_size_influence"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

env = RealEnv()
for dataset, value in all_res.items():

    # we also need to reconfigure the m for test
    dim = dim_map[dataset]
    update_m_with_dimension(dim)

    # print(f"[Info] Dataset {dataset}")
    with open(os.path.join(output_dir, dataset + ".log"), "w") as ofs:
        for res, percent in value:
            tput_arr = res[0]
            recall_arr = res[1]
            param_arr = res[2]

            # for non-full dataset, re-evaluate with full dataset
            if percent!=100:
                _tput = []
                _recall = []
                # print(f"[Info] Percentage {percent}")
                cnt = 1
                for param in param_arr:
                    print(f"[Info] Dataset {dataset}")
                    print(f"[Info] Percentage {percent}")
                    print(f"[Info] original-Param {cnt}: {param}")

                    # print(f"[Info] backward-Param {cnt}: {param}")
                    cnt += 1
                    env.config_system_with_params_original(param)
                    run_cmd = f'sudo timeout 1800 {RUN_ENGINE_PATH} "" "" {dataset}'

                    try:
                        result = sp.run(run_cmd, shell=True, stdout=sp.PIPE)
                        result = result.stdout.decode().split()

                        # Get the result
                        tput, recall, load_time, search_time = float(result[-2]), float(result[-3]), float(result[1]), float(result[2])
                    except:
                        tput, recall, load_time, search_time = 0, 0, 0, 0
                    
                    _tput.append(tput)
                    _recall.append(recall)
                
                tput_arr = _tput
                recall_arr = _recall
            else:
                print("Skip full dataset", dataset, percent)

            # output the result
            ofs.write(f"p{percent}-tput")
            for item in tput_arr:
                ofs.write(f"\t{item}")
            ofs.write(f"\np{percent}-recall")
            for item in recall_arr:
                ofs.write(f"\t{item}")
            ofs.write("\n")

