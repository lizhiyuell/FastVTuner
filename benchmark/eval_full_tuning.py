import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS_ROOT = PROJECT_ROOT / "systems"
for path in (PROJECT_ROOT, SYSTEMS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from systems.fastvtuner import FastVTunerSystem
from systems.vdtuner import VDTunerSystem


VDB_NAMES = ["milvus"]
DATASET_NAMES = ["gist"]
METHODS = [
    ("fastvtuner", FastVTunerSystem, 300),
    # ("vdtuner", VDTunerSystem, 300),
]


def run_method(method_name, system_cls, tuning_rounds, vdb_name, dataset_name):
    print(
        f"[start] method={method_name}, vdb={vdb_name}, dataset={dataset_name}, rounds={tuning_rounds}",
        flush=True,
    )
    system = system_cls(
        vdb_name=vdb_name,
        dataset_name=dataset_name,
    )
    for _ in range(tuning_rounds):
        system.step()
    print(
        f"[done] method={method_name}, vdb={vdb_name}, dataset={dataset_name}",
        flush=True,
    )


def main():
    for vdb_name in VDB_NAMES:
        for dataset_name in DATASET_NAMES:
            for method_name, system_cls, tuning_rounds in METHODS:
                run_method(method_name, system_cls, tuning_rounds, vdb_name, dataset_name)


if __name__ == "__main__":
    main()
