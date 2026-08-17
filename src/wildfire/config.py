from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "project.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)
