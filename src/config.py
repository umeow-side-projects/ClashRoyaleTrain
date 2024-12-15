import json

from typing import Any

class Config:
    filename = 'config.json'
    
    @staticmethod
    def get_config() -> dict:
        try:
            with open(Config.filename, 'r', encoding='utf-8') as f:
                return json.loads(f.read())
        except Exception:
            return {}
    
    @staticmethod
    def get(key) -> Any:
        try:
            config = Config.get_config()
            return config[key]
        except Exception:
            return None
    
    @staticmethod
    def check_key(key) -> bool:
        config = Config.get_config()
        return key in config
    
    @staticmethod
    def set(key, value) -> None:
        config = Config.get_config()
        config[key] = value
        
        with open(Config.filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(config, indent=4))