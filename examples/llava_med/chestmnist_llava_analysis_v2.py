"""
LLaVA-Med Analysis of ChestMNIST Dataset

This script applies LLaVA-Med to analyze ChestMNIST images for medical findings.
The pipeline includes:
1. Loading ChestMNIST dataset (64x64 grayscale chest X-rays)
2. Upsampling images to higher resolution for LLaVA-Med (configurable)
3. Converting grayscale images to RGB format for vision model
4. Applying LLaVA-Med for medical image analysis and label verification
5. Comparing LLaVA-Med predictions with ground truth labels

ChestMNIST contains 14 binary labels for multi-label classification:
- atelectasis, cardiomegaly, effusion, infiltration, mass, nodule, pneumonia,
- pneumothorax, consolidation, edema, emphysema, fibrosis, pleural, hernia
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import warnings

import gmi

# Suppress warnings
warnings.filterwarnings("ignore")

# Import our LLaVA-Med pipeline
import sys
sys.path.append('/workspace/gmi/examples/llava_med')
from llava_med import LLaVAMedPipeline

class ChestMNISTLLaVAMedAnalyzer:
    """
    Analyzer that applies LLaVA-Med to ChestMNIST images for medical analysis.
    """
    
    def __init__(self, 
                 upsampled_size: int = 128,
                 original_size: int = 64,
                 batch_size: int = 16,
                 device: Optional[str] = None):
        """
        Initialize the ChestMNIST LLaVA-Med analyzer.
        
        Args:
            upsampled_size: Target size for upsampling (default: 128)
            original_size: Original ChestMNIST size (default: 64)
            batch_size: Batch size for processing (default: 16)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.upsampled_size = upsampled_size
        self.original_size = original_size
        self.batch_size = batch_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # ChestMNIST label codebook
        self.chest_codebook = {
            0: 'atelectasis',
            1: 'cardiomegaly', 
            2: 'effusion',
            3: 'infiltration',
            4: 'mass',
            5: 'nodule',
            6: 'pneumonia',
            7: 'pneumothorax',
            8: 'consolidation',
            9: 'edema',
            10: 'emphysema',
            11: 'fibrosis',
            12: 'pleural',
            13: 'hernia'
        }
        
        # Medical condition descriptions for better LLaVA-Med prompts
        self.condition_descriptions = {
            'atelectasis': 'collapsed or incomplete expansion of lung tissue',
            'cardiomegaly': 'enlarged heart',
            'effusion': 'fluid in pleural space around lungs',
            'infiltration': 'abnormal substances in lung tissue',
            'mass': 'abnormal growth or tumor in chest',
            'nodule': 'small round growth in lungs',
            'pneumonia': 'infection causing inflammation in lungs',
            'pneumothorax': 'collapsed lung due to air in pleural space',
            'consolidation': 'lung tissue filled with liquid instead of air',
            'edema': 'excess fluid in lungs',
            'emphysema': 'damage to air sacs in lungs',
            'fibrosis': 'scarring and thickening of lung tissue',
            'pleural': 'conditions affecting pleura (lung lining)',
            'hernia': 'organ pushing through weak muscle or tissue'
        }
        
        # Setup data root
        self.data_root = '/workspace/gmi/gmi_data/datasets/medmnist_dataset_root/'
        os.makedirs(self.data_root, exist_ok=True)
        
        # Setup output directory
        self.output_dir = Path('/workspace/gmi/examples/llava_med/chestmnist_analysis')
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🔧 Initializing ChestMNIST LLaVA-Med Analyzer")
        print(f"   Original size: {original_size}x{original_size}")
        print(f"   Upsampled size: {upsampled_size}x{upsampled_size}")
        print(f"   Device: {self.device}")
        print(f"   Output directory: {self.output_dir}")
        
        # Initialize components
        self._load_datasets()
        self._setup_transforms()
        self._initialize_llava_med()
    
    def _load_datasets(self):
        """Load ChestMNIST datasets."""
        print("📥 Loading ChestMNIST datasets...")
        
        self.dataset_train = gmi.datasets.MedMNIST(
            'ChestMNIST',
            split='train',
            root=self.data_root,
            size=self.original_size,
            download=True
        )
        
        self.dataset_val = gmi.datasets.MedMNIST(
            'ChestMNIST',
            split='val', 
            root=self.data_root,
            size=self.original_size,
            download=True
        )
        
        self.dataset_test = gmi.datasets.MedMNIST(
            'ChestMNIST',
            split='test',
            root=self.data_root,
            size=self.original_size,
            download=True
        )
        
        print(f"   Train: {len(self.dataset_train)} samples")
        print(f"   Val: {len(self.dataset_val)} samples")
        print(f"   Test: {len(self.dataset_test)} samples")
        
        # Analyze sample
        sample_img, sample_label = self.dataset_train[0]
        print(f"   Sample image shape: {sample_img.shape}")
        print(f"   Sample label shape: {sample_label.shape}")
        print(f"   Image range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
    
    def _setup_transforms(self):
        """Setup image transforms for upsampling and format conversion."""
        print(f"🔄 Setting up transforms for {self.original_size}→{self.upsampled_size} upsampling")
        
        # Transform to upsample grayscale images to RGB format for LLaVA-Med
        self.transform = transforms.Compose([
            # Ensure tensor is in correct format and range
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1) if isinstance(x, torch.Tensor) else x),
            
            # Convert to PIL Image for transforms
            transforms.ToPILImage(mode='L'),  # Grayscale mode
            
            # Upsample using bilinear interpolation
            transforms.Resize(
                size=(self.upsampled_size, self.upsampled_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            ),
            
            # Convert to RGB (duplicate grayscale to 3 channels)
            transforms.Lambda(lambda img: img.convert('RGB')),
            
            # Convert back to tensor if needed
            transforms.ToTensor(),
        ])
        
        # Alternative transform that keeps as PIL Image for LLaVA-Med
        self.transform_pil = transforms.Compose([
            # Ensure tensor is in correct format and range
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1) if isinstance(x, torch.Tensor) else x),
            
            # Convert to PIL Image
            transforms.ToPILImage(mode='L'),
            
            # Upsample using bilinear interpolation
            transforms.Resize(
                size=(self.upsampled_size, self.upsampled_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            ),
            
            # Convert to RGB
            transforms.Lambda(lambda img: img.convert('RGB')),
        ])
    
    def _initialize_llava_med(self):
        """Initialize LLaVA-Med pipeline."""
        print("🤖 Initializing LLaVA-Med pipeline...")
        try:
            self.llava_pipeline = LLaVAMedPipeline(device=self.device)
            print("✅ LLaVA-Med pipeline initialized")
        except Exception as e:
            print(f"⚠️  LLaVA-Med initialization issue: {e}")
            print("   Continuing with framework demo mode")
            self.llava_pipeline = LLaVAMedPipeline(device=self.device)
    
    def process_single_image(self, 
                           image: torch.Tensor, 
                           label: torch.Tensor, 
                           idx: int) -> Dict:
        """
        Process a single ChestMNIST image with LLaVA-Med.
        
        Args:
            image: ChestMNIST image tensor [1, H, W]
            label: Multi-label binary tensor [14]
            idx: Sample index
            
        Returns:
            Dictionary with analysis results
        """
        # Convert to PIL Image for LLaVA-Med
        pil_image = self.transform_pil(image)
        
        # Get ground truth conditions
        gt_conditions = [self.chest_codebook[i] for i, val in enumerate(label) if val == 1]
        
        # Create comprehensive medical prompt
        base_prompt = (
            "You are analyzing a chest X-ray image. Please provide a detailed medical analysis including: "
            "1) Overall image quality and view, "
            "2) Anatomical structures visible, "
            "3) Any abnormalities or pathological findings, "
            "4) Specific conditions that may be present from this list: "
            f"{', '.join(self.chest_codebook.values())}. "
            "Please be specific about what you observe."
        )
        
        # Alternative focused prompt
        focused_prompt = (
            f"This is a chest X-ray. Look for signs of: {', '.join(self.chest_codebook.values())}. "
            "Which of these conditions, if any, are visible in this image? "
            "Explain your findings."
        )
        
        # Verification prompt based on ground truth
        if gt_conditions:
            verification_prompt = (
                f"This chest X-ray has been labeled with: {', '.join(gt_conditions)}. "
                "Do you see evidence of these conditions? Please analyze each one specifically."
            )
        else:
            verification_prompt = (
                "This chest X-ray has been labeled as normal (no pathological findings). "
                "Do you agree? What do you observe in this image?"
            )
        
        results = {
            'idx': idx,
            'gt_conditions': gt_conditions,
            'gt_labels': label.tolist(),
            'image_shape': pil_image.size,
            'analyses': {}
        }
        
        # Run multiple analyses with different prompts
        prompts = {
            'comprehensive': base_prompt,
            'focused': focused_prompt,
            'verification': verification_prompt
        }
        
        for prompt_name, prompt_text in prompts.items():
            try:
                print(f"   Running {prompt_name} analysis...")
                response = self.llava_pipeline.analyze_image(
                    image=pil_image,
                    question=prompt_text,
                    max_new_tokens=300,
                    temperature=0.1
                )
                results['analyses'][prompt_name] = {
                    'prompt': prompt_text,
                    'response': response
                }
            except Exception as e:
                print(f"   ⚠️  {prompt_name} analysis failed: {e}")
                results['analyses'][prompt_name] = {
                    'prompt': prompt_text,
                    'response': f"Analysis failed: {str(e)}"
                }
        
        return results
    
    def analyze_dataset_sample(self, 
                             dataset_split: str = 'test',
                             num_samples: int = 10,
                             start_idx: int = 0) -> List[Dict]:
        """
        Analyze a sample of images from the dataset.
        
        Args:
            dataset_split: Which split to use ('train', 'val', 'test')
            num_samples: Number of samples to analyze
            start_idx: Starting index in dataset
            
        Returns:
            List of analysis results
        """
        dataset = getattr(self, f'dataset_{dataset_split}')
        print(f"📊 Analyzing {num_samples} samples from {dataset_split} set (starting at {start_idx})")
        
        results = []
        
        for i in tqdm(range(start_idx, min(start_idx + num_samples, len(dataset))), 
                     desc=f"Analyzing {dataset_split} samples"):
            image, label = dataset[i]
            
            print(f"\\n🔍 Processing sample {i}")
            print(f"   Ground truth: {[self.chest_codebook[j] for j, val in enumerate(label) if val == 1]}")
            
            result = self.process_single_image(image, label, i)
            results.append(result)
            
            # Save intermediate results
            if (i - start_idx + 1) % 5 == 0:
                self._save_results(results, f"{dataset_split}_sample_{start_idx}_{i}.json")
        
        # Save final results
        self._save_results(results, f"{dataset_split}_analysis_{start_idx}_{start_idx + len(results) - 1}.json")
        
        return results
    
    def analyze_by_condition(self, 
                           condition: str,
                           dataset_split: str = 'test',
                           max_samples: int = 5) -> List[Dict]:
        """
        Analyze samples that have a specific condition.
        
        Args:
            condition: Condition name from chest_codebook
            dataset_split: Which dataset split to use
            max_samples: Maximum number of samples to analyze
            
        Returns:
            List of analysis results
        """
        if condition not in self.chest_codebook.values():
            raise ValueError(f"Condition '{condition}' not in codebook: {list(self.chest_codebook.values())}")
        
        # Find condition index
        condition_idx = [k for k, v in self.chest_codebook.items() if v == condition][0]
        
        dataset = getattr(self, f'dataset_{dataset_split}')
        print(f"🎯 Analyzing samples with condition: {condition} (index {condition_idx})")
        
        # Find samples with this condition
        matching_samples = []
        for i, (image, label) in enumerate(dataset):
            if label[condition_idx] == 1:
                matching_samples.append(i)
                if len(matching_samples) >= max_samples:
                    break
        
        print(f"   Found {len(matching_samples)} samples with {condition}")
        
        results = []
        for i in tqdm(matching_samples, desc=f"Analyzing {condition} samples"):
            image, label = dataset[i]
            print(f"\\n🔍 Processing {condition} sample {i}")
            
            result = self.process_single_image(image, label, i)
            results.append(result)
        
        # Save results
        self._save_results(results, f"{dataset_split}_{condition}_analysis.json")
        
        return results
    
    def _save_results(self, results: List[Dict], filename: str) -> None:
        """
        Save analysis results to JSON file.
        
        Args:
            results: List of analysis results
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        # Convert any numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            if isinstance(serializable_result.get('gt_labels'), np.ndarray):
                serializable_result['gt_labels'] = serializable_result['gt_labels'].tolist()
            serializable_results.append(serializable_result)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")


def main():
    """
    Main function to demonstrate ChestMNIST LLaVA-Med analysis.
    """
    print("🏥 ChestMNIST LLaVA-Med Analysis")
    print("=" * 50)
    
    # Configuration
    config = {
        'upsampled_size': 128,  # Can be changed to 256, 512, etc.
        'original_size': 64,
        'batch_size': 16,
        'device': None  # Auto-detect
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize analyzer
    analyzer = ChestMNISTLLaVAMedAnalyzer(**config)
    
    # Test 1: Analyze a few random samples
    print("\\n" + "=" * 50)
    print("TEST 1: Random Sample Analysis")
    print("=" * 50)
    
    random_results = analyzer.analyze_dataset_sample(
        dataset_split='test',
        num_samples=2,
        start_idx=0
    )
    
    # Test 2: Analyze samples with specific conditions
    print("\\n" + "=" * 50)
    print("TEST 2: Condition-Specific Analysis")
    print("=" * 50)
    
    # Analyze pneumonia cases
    try:
        pneumonia_results = analyzer.analyze_by_condition(
            condition='pneumonia',
            dataset_split='test',
            max_samples=1
        )
    except Exception as e:
        print(f"Pneumonia analysis failed: {e}")
        pneumonia_results = []
    
    print("\\n✅ ChestMNIST LLaVA-Med analysis completed!")
    print(f"   Results saved in: {analyzer.output_dir}")


if __name__ == "__main__":
    main()