import logging

import yaml

logger = logging.getLogger(__name__)


def load_config(file_path: str) -> dict:
    """Load YAML configuration file."""
    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config {file_path}: {str(e)}")
        raise
