from abc import ABC, abstractmethod
from pathlib import Path

# ------ base class abstraction ------
# The configuration class for different types of Vector databases

class VDBConfig(ABC):
    def __init__(self):
        self.base_path = Path(__file__).resolve().parent

    # write a certain configuration into the files
    @abstractmethod
    def write_configure(self, configurations):
        pass


# ------ Milvus ------
class VDBConfig_milvus(VDBConfig):
    def __init__(self):
        super().__init__()
        self.vdb_path = self.base_path / "milvus"
        self.config_path = self.base_path.parent / "vector-db-benchmark/experiments/configurations/milvus-single-node.json"
        self.config_path = self.vdb_path / 
    
    def write_configure(self, configurations):
        pass