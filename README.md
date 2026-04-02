# FastVTuner
Optimized VectorDB tuner based on VDTuner.


# Structure
- vector-db-benchmark: the main bencher for different vector databases
- systems: scripts for different tuning methods
- benchmark: directory for different benchers. Each subdirectory contains one scopes of benchmark.
- config: 直接命名为可能的参数及取值，current是当前参数及取值
- docker_config: 存储各种docker启动配置文件、系统级参数
- results: the directory to benchmark results

- systems/clients：adopted from vector-db-benchmark

# distance name
欧氏距离：euclidean
角度：angular
内积：inner-product

# TODO lists:
- base.py √
- vdb_config.py √