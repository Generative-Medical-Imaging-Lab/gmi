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
        
        # Handle output directory - use default 'outputs/' if None
        if self.config['output']['output_dir'] is None:
            # Get the directory where the config file is located (project root)
            project_root = os.path.dirname(os.path.abspath(config_path))
            self.config['output']['output_dir'] = os.path.join(project_root, 'outputs')
        
        # Create output directories
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
    def model(self) -> Dict[str, Any]:
        return self.config['model']
    
    @property
    def training(self) -> Dict[str, Any]:
        return self.config['training']
    
    @property
    def output(self) -> Dict[str, Any]:
        return self.config['output']
    
    @property
    def augmentation(self) -> Dict[str, Any]:
        return self.config['augmentation']
    
    @property
    def execution(self) -> Dict[str, Any]:
        return self.config['execution']

def parse_config(config_path: str) -> Config:
    """Parse configuration file and return Config object"""
    return Config(config_path)