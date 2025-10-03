import yaml
import torch
import os
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if torch.cuda.is_available() and 'cuda' in self.config['training']['device']:
            self.device = torch.device(self.config['training']['device'])
        else:
            self.device = torch.device('cpu')
        
        # Create output directories
        # If output_dir is null, default to script directory + outputs
        if self.config['output']['output_dir'] is None:
            script_dir = os.path.dirname(os.path.abspath(config_path))
            output_dir = os.path.join(script_dir, 'outputs')
            self.config['output']['output_dir'] = output_dir
        
        # Create output directories
        # If output_dir is null, default to script directory + outputs
        if self.config['output']['output_dir'] is None:
            script_dir = os.path.dirname(os.path.abspath(config_path))
            output_dir = os.path.join(script_dir, 'outputs')
            self.config['output']['output_dir'] = output_dir
        
        os.makedirs(self.config['output']['output_dir'], exist_ok=True)
        
        # Ensure dataset root exists
        os.makedirs(self.config['data']['dataset_root'], exist_ok=True)
    
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
    def physics(self) -> Dict[str, Any]:
        return self.config['physics']
    
    @property
    def model(self) -> Dict[str, Any]:
        return self.config['model']
    
    @property
    def training(self) -> Dict[str, Any]:
        return self.config['training']
    
    @property
    def output(self) -> Dict[str, Any]:
        return self.config['output']

def parse_config(config_path: str) -> Config:
    """Parse configuration file and return Config object"""
    return Config(config_path)