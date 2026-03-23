# FastVTuner
Optimized VectorDB tuner based on VDTuner.


# Structure
- vector-db-benchmark: the main bencher for different vector databases
- systems: scripts for different tuning methods
- benchmark: directory for different benchers. Each subdirectory contains one scopes of benchmark.
- config: configuration files for different Vector DB system. We define an abstract class, and have different realizations of different DB systems.
- results: the directory to benchmark results