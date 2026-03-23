# 给定配置，运行单次实验

from utils import *

dimension = 96
dataset = "deep-image-96-angular"
percentage = 10

# param_list = ['FLAT', 128, 7, 6, 8, 32, 256, 500, 500, 9905, 76.0, True, False, 88183, 7526631602, 7839]
param_list = ['SCANN', 2852, 44, 6, 8, 32, 256, 500, 733, 6765, 75.0, True, False, 55155, 9679515393, 3918]

param_list = [int(n) if isinstance(n, float) else n for n in param_list]

update_m_with_dimension(dimension)

# dataset_name = f"{dataset}-p-{percentage}"
dataset_name = f"{dataset}"

env = RealEnv()
env.config_system_with_params_original(param_list)
run_cmd = f'sudo timeout 1800 {RUN_ENGINE_PATH} "" "" {dataset_name}'

try:
    result = sp.run(run_cmd, shell=True, stdout=sp.PIPE)
    result = result.stdout.decode().split()

    # Get the result
    tput, recall, load_time, search_time = float(result[-2]), float(result[-3]), float(result[1]), float(result[2])
except:
    print("Finish with Error!!!")
    tput, recall, load_time, search_time = 0, 0, 0, 0


print(f"Exp finish, tput={tput}, recall={recall}, load_time={load_time}, search_time={search_time}")


