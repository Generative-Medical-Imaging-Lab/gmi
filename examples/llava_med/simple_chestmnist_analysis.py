"""
ChestMNIST LLaVA-Med Analysis - Simplified Version

This script applies LLaVA-Med to analyze ChestMNIST images, avoiding dependency conflicts.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
from typing import Dict, List, Optional
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import specific GMI components to avoid dependency issues
try:
    from gmi.datasets import MedMNIST
    print("✅ Successfully imported MedMNIST from GMI")
except ImportError as e:
    print(f"❌ Failed to import MedMNIST: {e}")
    print("   Please ensure GMI is properly installed")
    exit(1)

# Import our LLaVA-Med pipeline
import sys
sys.path.append('/workspace/gmi/examples/llava_med')

try:
    from llava_med import LLaVAMedPipeline
    print("✅ Successfully imported LLaVA-Med pipeline")
except ImportError as e:
    print(f"❌ Failed to import LLaVA-Med pipeline: {e}")
    exit(1)

class SimpleChestMNISTAnalyzer:
    """
    Simplified analyzer for ChestMNIST with LLaVA-Med.
    """
    
    def __init__(self, upsampled_size: int = 128, original_size: int = 64):
        self.upsampled_size = upsampled_size
        self.original_size = original_size
        
        # ChestMNIST label codebook
        self.chest_codebook = {
            0: 'atelectasis', 1: 'cardiomegaly', 2: 'effusion', 3: 'infiltration',
            4: 'mass', 5: 'nodule', 6: 'pneumonia', 7: 'pneumothorax',
            8: 'consolidation', 9: 'edema', 10: 'emphysema', 11: 'fibrosis',
            12: 'pleural', 13: 'hernia'
        }
        
        # Setup paths
        self.data_root = '/workspace/gmi/gmi_data/datasets/medmnist_dataset_root/'
        os.makedirs(self.data_root, exist_ok=True)
        
        self.output_dir = Path('/workspace/gmi/examples/llava_med/chestmnist_analysis')
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🔧 Initializing Simple ChestMNIST Analyzer")
        print(f"   Size: {original_size}x{original_size} → {upsampled_size}x{upsampled_size}")
        print(f"   Output: {self.output_dir}")
        
        self._setup_components()
    
    def _setup_components(self):
        """Setup datasets, transforms, and LLaVA-Med."""
        print("📥 Loading ChestMNIST dataset...")
        
        try:
            # Load test dataset
            self.dataset_test = MedMNIST(
                'ChestMNIST',
                split='test',
                root=self.data_root,
                size=self.original_size,
                download=True
            )
            print(f"   Test set: {len(self.dataset_test)} samples")
            
            # Check sample
            sample_img, sample_label = self.dataset_test[0]
            print(f"   Sample shape: {sample_img.shape}, label: {sample_label.shape}")
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            raise
        
        # Setup transforms
        print(f"🔄 Setting up {self.original_size}→{self.upsampled_size} transforms...")
        self.transform_pil = transforms.Compose([
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1) if isinstance(x, torch.Tensor) else x),
            transforms.ToPILImage(mode='L'),
            transforms.Resize(
                size=(self.upsampled_size, self.upsampled_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            ),
            transforms.Lambda(lambda img: img.convert('RGB')),
        ])
        
        # Initialize LLaVA-Med
        print("🤖 Initializing LLaVA-Med...")
        try:
            self.llava_pipeline = LLaVAMedPipeline()
            print("✅ LLaVA-Med ready")
        except Exception as e:
            print(f"⚠️  LLaVA-Med initialization: {e}")
            self.llava_pipeline = LLaVAMedPipeline()
    
    def analyze_sample(self, idx: int = 0) -> Dict:
        """Analyze a single sample."""
        print(f"\\n🔍 Analyzing sample {idx}")
        
        # Get sample
        image, label = self.dataset_test[idx]
        gt_conditions = [self.chest_codebook[i] for i, val in enumerate(label) if val == 1]
        
        print(f"   Ground truth: {gt_conditions or 'Normal'}")
        print(f"   Image shape: {image.shape}")
        
        # Transform image
        pil_image = self.transform_pil(image)
        print(f"   Upsampled to: {pil_image.size}")
        
        # Create prompt
        if gt_conditions:
            prompt = (
                f"This chest X-ray has been diagnosed with: {', '.join(gt_conditions)}. "
                "Do you see evidence of these conditions? Please analyze what you observe."
            )
        else:
            prompt = (
                "This chest X-ray appears normal. Please analyze this image and describe "
                "what you see. Are there any abnormalities or is it indeed normal?"
            )
        
        # Run analysis
        print("   Running LLaVA-Med analysis...")
        try:
            response = self.llava_pipeline.analyze_image(
                image=pil_image,
                question=prompt,
                max_new_tokens=200
            )
            
            result = {
                'idx': idx,
                'gt_conditions': gt_conditions,
                'gt_labels': label.tolist(),
                'image_shape': image.shape,
                'upsampled_size': pil_image.size,
                'prompt': prompt,
                'response': response
            }
            
            print(f"   ✅ Analysis completed")
            return result
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            return {
                'idx': idx,
                'gt_conditions': gt_conditions,
                'error': str(e)
            }
    
    def analyze_multiple_samples(self, indices: List[int]) -> List[Dict]:
        """Analyze multiple samples."""
        print(f"📊 Analyzing {len(indices)} samples...")
        
        results = []
        for idx in tqdm(indices, desc="Processing samples"):
            result = self.analyze_sample(idx)
            results.append(result)
            
            # Print summary
            if 'response' in result:
                print(f"\\nSample {idx}:")
                print(f"  GT: {result['gt_conditions'] or 'Normal'}")
                print(f"  Response: {result['response'][:100]}...")
        
        # Save results
        output_file = self.output_dir / f"analysis_samples_{min(indices)}_to_{max(indices)}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\\n💾 Results saved to: {output_file}")
        
        return results
    
    def find_samples_with_condition(self, condition: str, max_samples: int = 5) -> List[int]:
        """Find sample indices with a specific condition."""
        if condition not in self.chest_codebook.values():
            raise ValueError(f"Unknown condition: {condition}")
        
        condition_idx = [k for k, v in self.chest_codebook.items() if v == condition][0]
        
        matching_indices = []
        for i, (_, label) in enumerate(self.dataset_test):
            if label[condition_idx] == 1:
                matching_indices.append(i)
                if len(matching_indices) >= max_samples:
                    break
        
        print(f"🎯 Found {len(matching_indices)} samples with '{condition}': {matching_indices}")
        return matching_indices
    
    def create_visualization(self, results: List[Dict], save_path: Optional[str] = None):
        """Create visualization of results."""
        print("📊 Creating visualization...")
        
        fig, axes = plt.subplots(len(results), 3, figsize=(15, 5 * len(results)))
        if len(results) == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(results):
            if 'error' in result:
                axes[i, 0].text(0.5, 0.5, f"Error: {result['error']}", 
                               ha='center', va='center', transform=axes[i, 0].transAxes)
                axes[i, 0].set_title(f"Sample {result['idx']} - Error")
                continue
            
            # Original image
            idx = result['idx']
            image, label = self.dataset_test[idx]
            
            axes[i, 0].imshow(image.squeeze(), cmap='gray')
            axes[i, 0].set_title(f"Original {self.original_size}x{self.original_size}")
            axes[i, 0].axis('off')
            
            # Upsampled image
            upsampled = self.transform_pil(image)
            axes[i, 1].imshow(upsampled)
            axes[i, 1].set_title(f"Upsampled {self.upsampled_size}x{self.upsampled_size}")
            axes[i, 1].axis('off')
            
            # Analysis text
            axes[i, 2].axis('off')
            analysis_text = f"Sample {idx}\\n\\n"
            analysis_text += f"Ground Truth: {', '.join(result['gt_conditions']) or 'Normal'}\\n\\n"
            analysis_text += f"LLaVA-Med Analysis:\\n{result.get('response', 'No response')}"
            
            axes[i, 2].text(0.05, 0.95, analysis_text, 
                           ha='left', va='top', transform=axes[i, 2].transAxes,
                           fontsize=8, wrap=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved: {save_path}")
        
        plt.show()


def main():
    """Main demonstration function."""
    print("🏥 ChestMNIST LLaVA-Med Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SimpleChestMNISTAnalyzer(upsampled_size=128, original_size=64)
    
    # Test 1: Random samples
    print("\\n" + "=" * 30)
    print("TEST 1: Random Samples")
    print("=" * 30)
    
    random_indices = [0, 10, 50]
    random_results = analyzer.analyze_multiple_samples(random_indices)
    
    # Test 2: Specific condition
    print("\\n" + "=" * 30)  
    print("TEST 2: Pneumonia Cases")
    print("=" * 30)
    
    try:
        pneumonia_indices = analyzer.find_samples_with_condition('pneumonia', max_samples=2)
        if pneumonia_indices:
            pneumonia_results = analyzer.analyze_multiple_samples(pneumonia_indices)
        else:
            print("No pneumonia samples found")
            pneumonia_results = []
    except Exception as e:
        print(f"Error finding pneumonia samples: {e}")
        pneumonia_results = []
    
    # Test 3: Visualization
    print("\\n" + "=" * 30)
    print("TEST 3: Visualization")
    print("=" * 30)
    
    all_results = random_results + pneumonia_results
    if all_results:
        try:
            viz_path = analyzer.output_dir / "analysis_visualization.png"
            analyzer.create_visualization(all_results[:3], save_path=viz_path)
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    print("\\n✅ Analysis completed!")
    print(f"Results in: {analyzer.output_dir}")


if __name__ == "__main__":
    main()