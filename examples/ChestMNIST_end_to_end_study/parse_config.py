import yaml
import torch
import os
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if torch.cuda.is_available() and 'cuda' in self.config['device']:
            self.device = torch.device(self.config['device'])
        else:
            self.device = torch.device('cpu')
        
        # Create output directories
        os.makedirs(self.config['output']['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(self.config['output']['output_dir'], 
                                self.config['output']['visualization_dir']), exist_ok=True)
        
        # Ensure dataset root exists
        os.makedirs(self.config['data']['dataset_root'], exist_ok=True)
        
        # Convert relative paths to absolute paths for pretrained models
        script_dir = os.path.dirname(os.path.abspath(config_path))
        for key, path in self.config['pretrained_models'].items():
            if not os.path.isabs(path):
                self.config['pretrained_models'][key] = os.path.join(script_dir, path)
    
    def get(self, key_path: str, default=None):
        """Get nested config value using dot notation (e.g., 'data.batch_size')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @property
    def data(self) -> Dict[str, Any]:
        return self.config['data']
    
    @property
    def pretrained_models(self) -> Dict[str, Any]:
        return self.config['pretrained_models']
    
    @property
    def simulation(self) -> Dict[str, Any]:
        return self.config['simulation']
    
    @property
    def output(self) -> Dict[str, Any]:
        return self.config['output']

def parse_config(config_path: str) -> Config:
    """Parse configuration file and return Config object"""
    return Config(config_path)