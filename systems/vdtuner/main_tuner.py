import sys 
sys.path.append("..") 

import os
from optimizer_pobo_sa import PollingBayesianOptimization
from utils import RealEnv
import json

INDEX_PARAM_PATH = '/root/lzy/FVDB_tuning/VDTuner/auto-configure/index_param.json'
KNOB_PATH = r'/root/lzy/FVDB_tuning/VDTuner/auto-configure/whole_param.json'

# 更新m，保证是dimension的因子
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

if __name__ == '__main__':

    # run multiple tests simultaneously
    total_nr_step = 300
    # total_nr_step = 8

    result_dir = "./results"
    # result_dir = "./result_temp"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # dataset name, start_pos, end_pos, test_idx, dimension
    # run_params = [
    #     ["deep-image-96-angular", 0, 100, 0, 96],
    #     ["gist-960-euclidean", 0, 100, 0, 960],
    #     ["glove-100-angular", 0, 100, 0, 100],
    # ]

    # # 使用采样后的数据集测试
    run_params = [
        ["glove-100-angular-p-10", 0, 100, 0, 100],
        ["glove-100-angular-p-30", 0, 100, 0, 100],
        ["glove-100-angular-p-50", 0, 100, 0, 100],
        ["glove-100-angular-p-70", 0, 100, 0, 100],
        ["glove-100-angular-p-90", 0, 100, 0, 100],
        ["glove-100-angular-p-100", 0, 100, 0, 100],
        ["gist-960-euclidean-p-10", 0, 100, 0, 960],
        ["gist-960-euclidean-p-30", 0, 100, 0, 960],
        ["gist-960-euclidean-p-50", 0, 100, 0, 960],
        ["gist-960-euclidean-p-70", 0, 100, 0, 960],
        ["gist-960-euclidean-p-90", 0, 100, 0, 960],
        ["gist-960-euclidean-p-100", 0, 100, 0, 960],
        ["deep-image-96-angular-p-10", 0, 100, 0, 96],
        ["deep-image-96-angular-p-30", 0, 100, 0, 96],
        ["deep-image-96-angular-p-50", 0, 100, 0, 96],
        ["deep-image-96-angular-p-70", 0, 100, 0, 96],
        ["deep-image-96-angular-p-90", 0, 100, 0, 96],
        ["deep-image-96-angular-p-100", 0, 100, 0, 96],
    ]

    # benchmark of each params
    for param in run_params:

        # 根据向量维度，调节可能的m取值为dimension的所有可能的因子
        dimension = param[4]
        update_m_with_dimension(dimension)

        run_cmd_param = f'"" "" {param[0]} "" "" {param[1]} {param[2]}'
        record_file_name = f"record_{param[0]}_start_{param[1]}_end_{param[2]}_round_{param[3]}.log"
        pobo_record_file_name = f"pobo_{record_file_name}"

        # put the results into the result dir
        record_file_name = os.path.join(result_dir, record_file_name)
        pobo_record_file_name = os.path.join(result_dir, pobo_record_file_name)

        # remove any previous result log
        if os.path.isfile(record_file_name):
            os.remove(record_file_name)
        if os.path.isfile(pobo_record_file_name):
            os.remove(pobo_record_file_name)

        # prepare the environment
        env = RealEnv()
        model = PollingBayesianOptimization(env, seed=1, run_params=run_cmd_param, log_file=record_file_name, pobo_file=pobo_record_file_name)
        
        # initial sampling
        model.init_sample()

        # iterative auto-tuning
        for i in range(total_nr_step-7):
            model.step()
