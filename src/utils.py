import os, json
import numpy as np
import tensorflow as tf
import yaml

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
