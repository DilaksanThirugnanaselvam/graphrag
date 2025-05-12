import yaml
import os

def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML file.
        
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_to_file(content: str, path: str):
    """
    Save content to a file.
    
    Args:
        content (str): Content to save.
        path (str): File path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
