import yaml
import os
from pathlib import Path
from typing import Dict, Any
from functools import lru_cache

@lru_cache()
def get_config() -> Dict[str, Any]:
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables
    config['admin_token'] = os.getenv('ADMIN_TOKEN', 'change-me')
    config['strict_blocklist'] = os.getenv('STRICT_BLOCKLIST', 'true').lower() == 'true'
    config['ollama_host'] = os.getenv('OLLAMA_HOST', None)
    config['log_level'] = os.getenv('LOG_LEVEL', 'INFO')
    
    return config

def get_storage_path() -> Path:
    return Path(__file__).parent / "storage"

def ensure_storage_dirs():
    storage_path = get_storage_path()
    storage_path.mkdir(exist_ok=True)
    (storage_path / "chroma").mkdir(exist_ok=True)