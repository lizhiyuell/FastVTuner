import sys 
sys.path.append("..") 

import joblib
from scipy.stats import qmc
import json
import numpy as np
import time
import subprocess as sp
import random
from configure import *
import os
import re

KNOB_PATH = r'/root/lzy/FVDB_tuning/VDTuner/auto-configure/whole_param.json'
RUN_ENGINE_PATH = r'/root/lzy/FVDB_tuning/VDTuner/vector-db-benchmark-master/run_engine.sh'
# the same with main tuner
INDEX_PARAM_PATH = r'/root/lzy/FVDB_tuning/VDTuner/auto-configure/index_param.json'

# 拉丁超立方采样，它在每个维度上只生成[0, 1]区间的点作为拉丁采样，后续的Knob函数会负责将这个归一化的采样拓展到具体函数中
def LHS_sample(dimension, num_points, seed):
    sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    latin_samples = sampler.random(n=num_points)

    # 返回矩阵：每一行为一个采样点，每一列是一个采样点在各个维度上的值
    return latin_samples

# 这个参数就是加载所有可能的参数，然后将它们映射到[0, 1]，或者反向映射
class KnobStand:
    def __init__(self, path) -> None:
        self.path = path
        with open(path, 'r') as f:
            # 这个json文件记录了所有可调的参数、类型以及它们的取值范围，可用类型就两个：整型和枚举类型
            self.knobs_detail = json.load(f)

    def scale_back(self, knob_name, zero_one_val):
        knob = self.knobs_detail[knob_name]
        if knob['type'] == 'integer':
            real_val = zero_one_val * (knob['max'] - knob['min']) + knob['min']
            return int(real_val), int(real_val)

        elif knob['type'] == 'enum':
            enum_size = len(knob['enum_values'])
            enum_index = int(enum_size * zero_one_val)
            enum_index = min(enum_size - 1, enum_index)
            real_val = knob['enum_values'][enum_index]
            return enum_index, real_val
    
    def scale_forward(self, knob_name, real_val):
        knob = self.knobs_detail[knob_name]
        if knob['type'] == 'integer':
            zero_one_val = (real_val - knob['min']) / (knob['max'] - knob['min'])
            return zero_one_val

        elif knob['type'] == 'enum':
            enum_size = len(knob['enum_values'])
            zero_one_val = knob['enum_values'].index(real_val) / enum_size
            return zero_one_val

class StaticEnv:
    def __init__(self, model_path=['XGBoost_20knob_thro.model', 'XGBoost_20knob_prec.model'], knob_path=r'milvus_important_params.json') -> None:
        self.model_path = model_path
        self.get_surrogate(model_path)
        self.knob_stand = KnobStand(knob_path)
        self.names = list(self.knob_stand.knobs_detail.keys())
        self.t1 = time.time()
        self.sampled_times = 0

        self.X_record = []
        self.Y1_record = []
        self.Y2_record = []
        self.Y_record = []

    def get_surrogate(self, surrogate_path):
        # surrogate1, surrogate2 = joblib.load(surrogate_path[0]), joblib.load(surrogate_path[1])
        self.model1, self.model2 = joblib.load(surrogate_path[0]), joblib.load(surrogate_path[1])

    def get_state(self, knob_vals_arr):
        Y1, Y2 = [], []
        for i,record in enumerate(knob_vals_arr):
            conf_value = [self.knob_stand.scale_back(self.names[j], knob_val)[0] for j,knob_val in enumerate(record)]
            print(f"Index parameters changed: {conf_value}")

            y1 = self.model1.predict([conf_value])[0]
            y2 = self.model2.predict([conf_value])[0]

            self.sampled_times += 1
            print(f'[{self.sampled_times}] {int(time.time()-self.t1)} {y1} {y2}')
            
            Y1.append(y1)
            Y2.append(y2)
        return np.concatenate((np.array(Y1).reshape(-1,1), np.array(Y2).reshape(-1,1)), axis=1)

class RealEnv:
    def __init__(self, bench_path=RUN_ENGINE_PATH, knob_path=KNOB_PATH) -> None:
        self.bench_path = bench_path
        self.knob_stand = KnobStand(knob_path)
        # 所有可以调节的目标名称
        self.names = list(self.knob_stand.knobs_detail.keys())
        self.t1 = time.time()
        self.t2 = time.time()
        self.sampled_times = 0

        self.X_record = []
        self.Y1_record = []
        self.Y2_record = []
        self.Y_record = []

    # 这个knob_vals_arr应该包含了多个要调节的参数值（也合理，因为一次计算会有多条历史观测记录）
    def get_state(self, knob_vals_arr, run_params, log_file):
        Y1, Y2, Y3 = [], [], []
        for i,record in enumerate(knob_vals_arr):
            # 按照论文的说法，每一轮调参的时候，会把所有参数都调一遍，这样的话record也会包含所有的参数及其设定值了
            # 这一步将原始的设定值映射为了[0, 1]中的数
            conf_value = [self.knob_stand.scale_back(self.names[j], knob_val)[1] for j,knob_val in enumerate(record)]
            
            # 这里应该是硬指定了索引的参数和系统的参数
            index_value, system_value = conf_value[:9], conf_value[9:]
            index_name, system_name = self.names[:9], self.names[9:]

            # 具体的索引参数值和系统参数值，并进行config
            index_conf = dict(zip(index_name,index_value))
            system_conf = dict(zip(system_name,system_value))

            # 这里会把对应的配置参数写到配置文件中
            configure_index(*filter_index_rule(index_conf))
            configure_system(filter_system_rule(system_conf))

            # print(f"Parameters changed to: {index_conf} {system_conf}")

            # 配置文件写好后，执行单次测试运行
            try:
                print("----------")
                print(index_conf)
                print(system_conf)
                start_time = time.time()
                # result = sp.run(f'sudo timeout 900 {RUN_ENGINE_PATH} "" "" glove-100-angular', shell=True, stdout=sp.PIPE)
                cmd = f'sudo timeout 1800 {RUN_ENGINE_PATH} {run_params}'
                print(f"[Run CMD] {cmd}")
                result = sp.run(cmd, shell=True, stdout=sp.PIPE)
                end_time = time.time()
                print(f"[One round] {end_time-start_time}s")
                result = result.stdout.decode().split()
                y1, y2 = float(result[-2]), float(result[-3])

                upload_time, total_load_time = float(result[0]), float(result[1])
                search_time = float(result[2])
                
                # 应该分别对应Tput和Recall
                self.Y1_record.append(y1)
                self.Y2_record.append(y2)
            except:
                y1, y2 = min(self.Y1_record), min(self.Y2_record)
                upload_time = 0
                total_load_time = 0
                search_time = 0
            
            # y3是单轮运行的时间
            y3 = int(time.time()-self.t2)
            self.sampled_times += 1

            self.t2 = time.time()
            print(f'[{self.sampled_times}] {int(self.t2-self.t1)} {y1} {y2} {y3}')
            sp.run(f'echo [{self.sampled_times}] {int(self.t2-self.t1)} {index_conf} {system_conf} {upload_time} {total_load_time} {search_time} {y1} {y2} {y3} >> {log_file}', shell=True, stdout=sp.PIPE)

            Y1.append(y1)
            Y2.append(y2)
            Y3.append(y3)

        return np.array([Y1,Y2,Y3]).T

    # 将参数配置到系统中
    def config_system_with_params(self, knob_vals_arr):

        conf_value = [self.knob_stand.scale_back(self.names[j], knob_val)[1] for j,knob_val in enumerate(knob_vals_arr)]

        index_value, system_value = conf_value[:9], conf_value[9:]
        index_name, system_name = self.names[:9], self.names[9:]

        index_conf = dict(zip(index_name,index_value))
        system_conf = dict(zip(system_name,system_value))

        # 将系统参数进行配置
        configure_index(*filter_index_rule(index_conf))
        configure_system(filter_system_rule(system_conf))

    # 一个新的配置函数，直接把原始参数进行config，不再做scale
    def config_system_with_params_original(self, conf_value):

        index_value, system_value = conf_value[:9], conf_value[9:]
        index_name, system_name = self.names[:9], self.names[9:]

        index_conf = dict(zip(index_name,index_value))
        system_conf = dict(zip(system_name,system_value))

        # 将系统参数进行配置
        configure_index(*filter_index_rule(index_conf))
        configure_system(filter_system_rule(system_conf))


    def default_conf(self):
        return [self.knob_stand.scale_forward(k, v['default']) for k,v in self.knob_stand.knobs_detail.items()]

### --------- functions added by us --------- ###
# the parameter 'm' must be 
def update_m_with_dimension(dimension):
    # 1. find all factors of 'dimension' and sort them
    factors = []
    for i in range(1, dimension + 1):
        if dimension % i == 0:
            factors.append(i)
    factors.sort()

    # 2. find the one that is nearest to '10' as the default value
    default_value = min(factors, key=lambda x: abs(x - 10))

    # 3. read the index_param.json and whole_param.json, updating m.default and m.enum_values
    for filename in [INDEX_PARAM_PATH, KNOB_PATH]:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["m"]["default"] = default_value
        data["m"]["enum_values"] = factors

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)



BRACE_RE = re.compile(r"\{([^{}]*)\}")
NUM_RE = re.compile(r"^\d+(\.\d+)?$")

def cast_value(s: str):
    s = s.strip()
    if s == "True":
        return True
    if s == "False":
        return False
    if NUM_RE.match(s):
        temp_num = float(s)
        if temp_num < 1:
            return temp_num
        else:
            return int(temp_num)
    return s

def _convert_value(value_str: str):
    s = value_str.strip()

    # Bool
    if s == "True":
        return True
    if s == "False":
        return False

    # Number (int/float)
    if _NUM_RE.match(s):
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        return float(s)

    # Fallback: string
    return s

# extract the parameters from a line, and return it in the uniform format
def extract_line_params(line):

    # may change the env for different databases
    env = RealEnv()
    knob_names = env.names
    knob_num = len(knob_names)

    # init the knob and fill it with the value
    x = [0 for _ in range(knob_num)]

    cur_idx = 0

    # extract all parameters and scale forward
    values = []
    for m in BRACE_RE.finditer(line):
        inner = m.group(1)

        for part in inner.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                continue
            k, v = part.split(":", 1)
            # specially handle for certain key
            original_val = cast_value(v)
            if k=="dataCoord*segment*sealProportion":
                original_val = original_val * 100
            # # scale forward
            # val = env.knob_stand.scale_forward(knob_names[cur_idx], original_val)
            # values.append(val)
            values.append(original_val)
            cur_idx += 1

    return values


# for a given result file, get the Pareto result of it
def get_Pareto_result(filename):

    # [tput, recall, {param_list}]
    res = []

    with open(filename, "r") as ifs:
        for line in ifs:
            content = line.split()
            tput = float(content[-3])
            recall = float(content[-2])
            res.append([tput, recall, extract_line_params(line)])

    res = sorted(res, key=lambda x : x[0])

    final = [[], [], []]
    max_recall = 0
    pos = len(res)-1
    while pos >= 0:
        t, r, p = res[pos][0], res[pos][1], res[pos][2]
        if r > max_recall:
            final[0].append(t)
            final[1].append(r)
            final[2].append(p)
        max_recall = max(max_recall, r)
        pos -= 1

    final[0].reverse()
    final[1].reverse()
    final[2].reverse()

    return final


if __name__ == '__main__':
    print(type(LHS_sample(5,10)))

