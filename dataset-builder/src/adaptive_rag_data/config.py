
from dataclasses import dataclass
from typing import Any, Dict
import yaml


@dataclass
class Config:
    data: Dict[str, Any]

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return Config(cfg)

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self.data
        for k in keys:
            if k not in node:
                return default
            node = node[k]
        return node
