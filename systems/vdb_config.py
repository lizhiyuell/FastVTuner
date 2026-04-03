from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from common import *

SUPPORTED_VDBS = {"milvus", "ex-milvus", "qudrant", "ex-qudrant"}

CURRENT_CONFIG_PATH = CONFIG_ROOT / "current.json"

class VDBConfig:
    def __init__(
        self,
        vdb_name: str
    ) -> None:
        if vdb_name not in SUPPORTED_VDBS:
            raise ValueError(f"Unsupported vdb_name: {vdb_name}")

        self.vdb_name = vdb_name

        # the template for all tunable parameters
        self.config_template_path = CONFIG_ROOT / f"{vdb_name}.json"
        self.config_current_path = CURRENT_CONFIG_PATH
        
        # set the system parameter for the milvus
        if "milvus" in vdb_name:
            self.config_system_path = DOCKER_CONFIG_ROOT / "milvus" / "milvus.yaml"

        # loading the parameter information from the file
        with self.config_template_path.open("r", encoding="utf-8") as f:
            meta_config = json.load(f)

        # parsing the parameter information
        self.param_names = list(meta_config.keys())
        # default values in the same order as param_names
        self.default_config = []
        # save the normalized paramters between [0, 1]
        self.normalized_parameter = []
        # unormalized parameters
        self.current_params = []
        # the description metadata of the parameters
        self.param_meta = []

        # load the parameter meta data and default values
        for param_name in self.param_names:
            detail = meta_config[param_name]

            # load the default parameters
            self.default_config.append(detail["default"])
            self.current_params.append(detail["default"])

            knob_type = detail["type"]
            knob_class = detail.get("class")
            meta: dict[str, Any] = {}
            meta["name"] = param_name
            meta["type"] = knob_type
            meta["class"] = knob_class
            if knob_type == "integer":
                meta["min"] = detail["min"]
                meta["max"] = detail["max"]
            elif knob_type == "float":
                meta["min"] = float(detail["min"])
                meta["max"] = float(detail["max"])
            elif knob_type == "enum":
                enum_values = list(detail["enum_values"])
                meta["min"] = 0
                meta["max"] = len(enum_values) - 1
                meta["enum_values"] = enum_values
            else:
                raise ValueError(f"Unsupported knob type: {knob_type}")
            
            self.param_meta.append(meta)

        self.param_normalized()

    def param_normalized(self) -> None:
        self.normalized_parameter = []
        for meta, param_value in zip(self.param_meta, self.current_params):
            if meta["type"] == "integer":
                self.normalized_parameter.append(
                    (param_value - meta["min"]) / (meta["max"] - meta["min"])
                )
            elif meta["type"] == "float":
                self.normalized_parameter.append(
                    (float(param_value) - float(meta["min"]))
                    / (float(meta["max"]) - float(meta["min"]))
                )
            elif meta["type"] == "enum":
                self.normalized_parameter.append(
                    meta["enum_values"].index(param_value) / len(meta["enum_values"])
                )
            else:
                raise ValueError(f"Unsupported knob type: {meta['type']}")

    def param_original(self) -> None:
        self.current_params = []
        for meta, param_value in zip(self.param_meta, self.normalized_parameter):
            if meta["type"] == "integer":
                self.current_params.append(
                    int(float(param_value) * (meta["max"] - meta["min"]) + meta["min"])
                )
            elif meta["type"] == "float":
                self.current_params.append(
                    float(param_value) * (float(meta["max"]) - float(meta["min"]))
                    + float(meta["min"])
                )
            elif meta["type"] == "enum":
                enum_size = len(meta["enum_values"])
                enum_index = int(enum_size * float(param_value))
                enum_index = min(enum_size - 1, enum_index)
                self.current_params.append(meta["enum_values"][enum_index])
            else:
                raise ValueError(f"Unsupported knob type: {meta['type']}")

    def _get_param_meta(self, param_name):
        for meta in self.param_meta:
            if meta["name"] == param_name:
                return meta
        raise ValueError(f"Unknown parameter name: {param_name}")

    def get_normalized_param(self) -> list[float]:
        return list(self.normalized_parameter)

    def get_original_param(self) -> list[Any]:
        return list(self.current_params)

    def get_param_index(self, param_name):
        return self.param_names.index(param_name)

    def get_normalized(self, param_name, param_value):
        meta = self._get_param_meta(param_name)

        if meta["type"] == "integer":
            return (param_value - meta["min"]) / (meta["max"] - meta["min"])
        if meta["type"] == "float":
            return (
                (float(param_value) - float(meta["min"]))
                / (float(meta["max"]) - float(meta["min"]))
            )
        if meta["type"] == "enum":
            return meta["enum_values"].index(param_value) / len(meta["enum_values"])

        raise ValueError(f"Unsupported knob type: {meta['type']}")

    def get_original(self, param_name, param_value):
        meta = self._get_param_meta(param_name)

        if meta["type"] == "integer":
            return int(float(param_value) * (meta["max"] - meta["min"]) + meta["min"])
        if meta["type"] == "float":
            return (
                float(param_value) * (float(meta["max"]) - float(meta["min"]))
                + float(meta["min"])
            )
        if meta["type"] == "enum":
            enum_size = len(meta["enum_values"])
            enum_index = int(enum_size * float(param_value))
            enum_index = min(enum_size - 1, enum_index)
            return meta["enum_values"][enum_index]

        raise ValueError(f"Unsupported knob type: {meta['type']}")

    def set_normalized_param(self, params: list[float], apply = True) -> None:
        if len(params) != len(self.param_names):
            raise ValueError(
                f"Parameter length mismatch: expected {len(self.param_names)}, got {len(params)}"
            )

        for meta, value in zip(self.param_meta, params):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"Normalized parameter `{meta['name']}` out of range: {value}, expected [0, 1]"
                )

        self.normalized_parameter = list(params)
        self.param_original()

        if apply:
            self.apply_params()

    def set_original_param(self, params: list[Any], apply=True) -> None:
        if len(params) != len(self.param_names):
            raise ValueError(
                f"Parameter length mismatch: expected {len(self.param_names)}, got {len(params)}"
            )

        for meta, value in zip(self.param_meta, params):
            if meta["type"] == "enum":
                if value not in meta["enum_values"]:
                    raise ValueError(
                        f"Original parameter `{meta['name']}` out of range: {value}, "
                        f"expected one of {meta['enum_values']}"
                    )
                continue

            numeric_value = float(value)
            if not meta["min"] <= numeric_value <= meta["max"]:
                raise ValueError(
                    f"Original parameter `{meta['name']}` out of range: {value}, "
                    f"expected [{meta['min']}, {meta['max']}]"
                )

        self.current_params = list(params)
        self.param_normalized()

        if apply:
            self.apply_params()

    # split the original "*" style parameters into the key-value style
    def _set_nested_value(self, tree: dict[str, Any], path: str, value: Any) -> None:
        keys = path.split("*")
        node = tree
        for key in keys[:-1]:
            if key not in node:
                raise KeyError(f"Key path `{path}` not found: missing `{key}`")
            if not isinstance(node[key], dict):
                raise TypeError(f"Key path `{path}` is invalid: `{key}` is not a dict")
            node = node[key]
        node[keys[-1]] = value

    def _split_current_params(self) -> tuple[dict[str, Any], dict[str, Any]]:
        index_conf: dict[str, Any] = {}
        system_conf: dict[str, Any] = {}
        for meta, value in zip(self.param_meta, self.current_params):
            if meta["class"] == "system":
                system_conf[meta["name"]] = value
            else:
                index_conf[meta["name"]] = value
        return index_conf, system_conf

    def _milvus_apply_index_config(self, index_conf: dict[str, Any]) -> None:
        index_type = None
        building_params: dict[str, Any] = {}
        searching_params: dict[str, Any] = {}
        for meta, value in zip(self.param_meta, self.current_params):
            if meta["class"] == "type":
                index_type = value
            elif meta["class"] == "building":
                building_params[meta["name"]] = value
            elif meta["class"] == "searching":
                searching_params[meta["name"]] = value

        if index_type is None:
            raise ValueError("Missing index_type in current params")

        config = {
            "name": "tuning-config",
            "engine": "milvus",
            "index": index_type,
            "connection_params": {},
            "collection_params": {},
            "search_params": {
                "parallel": 1,
                "config": dict(searching_params),
            },
            "upload_params": {
                "parallel": 1,
                "index_params": dict(building_params),
            },
        }

        with self.config_current_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _milvus_apply_system_config(self, system_conf: dict[str, Any]) -> None:
        with self.config_system_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for key, value in system_conf.items():
            self._set_nested_value(config, key, value)

        with self.config_system_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    def apply_params(self) -> None:
        if self.vdb_name == "milvus":
            # split the parameters into index paramters and system paramters
            index_conf, system_conf = self._split_current_params()
            self._milvus_apply_index_config(index_conf)
            self._milvus_apply_system_config(system_conf)
        else:
            raise NotImplementedError(f"{self.vdb_name} config is not implemented yet")


if __name__ == "__main__":
    vdb_config = VDBConfig("milvus")

    default_params = dict(zip(vdb_config.param_names, vdb_config.get_original_param()))
    override_params = {
        "index_type": "HNSW",
        "M": 32,
        "efConstruction": 200,
        "ef": 64,
    }
    default_params.update(override_params)

    vdb_config.set_original_param(
        [default_params[name] for name in vdb_config.param_names]
    )
    vdb_config.apply_params()
