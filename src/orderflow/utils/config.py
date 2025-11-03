"""
config.py
----------
Utility to load and validate the global configuration file (YAML).
Every module in the project can import `get_config()` to access paths, symbols, model params, etc.
"""

import os
import yaml
from pydantic import BaseModel, Field
from typing import List, Dict, Any


# ---------- Pydantic models for type safety ----------

class ModelConfig(BaseModel):
    type: str = Field(..., description="Model type, e.g., xgboost or logistic_regression")
    params: Dict[str, Any] = Field(default_factory=dict)


class PathsConfig(BaseModel):
    raw: str
    interim: str
    processed: str
    models: str
    reports: str


class Settings(BaseModel):
    storage: Dict[str, str]
    symbols: List[str]
    horizons: List[str]
    frequency: str
    data_sources: Dict[str, str]
    model: ModelConfig
    paths: PathsConfig


# ---------- Loader function ----------

def get_config(path: str = "configs/settings.yaml") -> Settings:
    """
    Loads the YAML configuration and returns a validated Settings object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    config = Settings(**cfg_dict)

    # Ensure output directories exist
    for key, folder in config.paths.dict().items():
        os.makedirs(folder, exist_ok=True)

    return config


# Example usage for testing
if __name__ == "__main__":
    cfg = get_config()
    print("Tracked symbols:", cfg.symbols)
    print("Data frequency:", cfg.frequency)
