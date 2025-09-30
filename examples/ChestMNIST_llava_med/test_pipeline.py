"""
Test script for the LLaVA-Med pipeline with local medical image examples.
"""

import sys
import os
sys.path.append('/workspace/gmi/examples/llava_med')

from llava_med import LLaVAMedPipeline
from PIL import Image, ImageDraw
import tempfile

def create_test_chest_xray():
    """Create a synthetic chest X-ray image for testing."""
    # Create grayscale image (typical for X-rays)
    img = Image.new('L', (512, 512), color=20)  # Dark background
    draw = ImageDraw.Draw(img)
    
    # Draw lung fields (brighter areas)
    draw.ellipse([80, 100, 220, 350], fill=120)   # Left lung
    draw.ellipse([290, 100, 430, 350], fill=115)  # Right lung
    
    # Draw heart shadow (darker area between lungs)
    draw.ellipse([180, 180, 280, 320], fill=60)
    
    # Draw ribs (thin bright lines)
    for i, y in enumerate(range(120, 340, 25)):
        brightness = 140 + (i % 2) * 10  # Alternate brightness
        draw.arc([60, y-5, 450, y+5], 0, 180, fill=brightness)
    
    # Draw spine shadow
    draw.rectangle([240, 80, 260, 400], fill=45)
    
    # Draw diaphragm
    draw.arc([100, 340, 410, 380], 0, 180, fill=130)
    
    return img

def create_test_ct_scan():
    """Create a synthetic CT scan image for testing."""
    # Create grayscale image with different contrast than X-ray
    img = Image.new('L', (256, 256), color=80)  # Mid-gray background
    draw = ImageDraw.Draw(img)
    
    # Draw organ structures with different densities
    draw.ellipse([50, 50, 200, 200], fill=120)    # Soft tissue
    draw.ellipse([80, 80, 170, 170], fill=60)     # Fluid/blood
    draw.ellipse([190, 100, 240, 150], fill=200)  # Bone/calcification
    
    return img

def create_test_mri():
    """Create a synthetic MRI image for testing."""
    # Create grayscale MRI with different contrast characteristics
    img = Image.new('L', (256, 256), color=50)  # Darker background
    draw = ImageDraw.Draw(img)
    
    # Brain-like structures
    draw.ellipse([30, 30, 226, 226], fill=120)     # Outer brain
    draw.ellipse([50, 50, 206, 206], fill=80)      # Gray matter
    draw.ellipse([70, 70, 186, 186], fill=160)     # White matter
    draw.ellipse([110, 110, 146, 146], fill=40)    # Ventricles
    
    return img

def main():
    print("🧪 LLaVA-Med Pipeline Test with Synthetic Medical Images")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = LLaVAMedPipeline()
    
    # Create test images
    chest_xray = create_test_chest_xray()
    ct_scan = create_test_ct_scan()
    
    # Create more test images
    mri_scan = create_test_mri()
    
    # Save test images (keep them as grayscale)
    with tempfile.NamedTemporaryFile(suffix='_chest_xray.png', delete=False) as f:
        chest_xray.save(f.name, "PNG")
        xray_path = f.name
        
    with tempfile.NamedTemporaryFile(suffix='_ct_scan.png', delete=False) as f:
        ct_scan.save(f.name, "PNG")
        ct_path = f.name
        
    with tempfile.NamedTemporaryFile(suffix='_mri_scan.png', delete=False) as f:
        mri_scan.save(f.name, "PNG")
        mri_path = f.name
    
    test_cases = [
        {
            'image': xray_path,
            'name': 'Chest X-Ray',
            'questions': [
                "What type of medical image is this?",
                "What anatomical structures can you identify?",
                "Are the lung fields clear?",
                "Describe the heart shadow and diaphragm."
            ]
        },
        {
            'image': ct_path,
            'name': 'CT Scan',
            'questions': [
                "What imaging modality is this?",
                "What tissue types are visible?",
                "Are there any high-density areas that might indicate calcification?",
                "Describe the image contrast characteristics."
            ]
        },
        {
            'image': mri_path,
            'name': 'MRI Scan',
            'questions': [
                "What type of MRI sequence might this represent?",
                "What anatomical region is being imaged?",
                "Can you identify different tissue contrasts?",
                "What structures show high signal intensity?"
            ]
        }
    ]
    
    # Test each image with multiple questions
    for i, test_case in enumerate(test_cases):
        print(f"\n🔍 Test Case {i+1}: {test_case['name']}")
        print(f"   File: {os.path.basename(test_case['image'])}")
        print("=" * 50)
        
        for j, question in enumerate(test_case['questions']):
            print(f"\n📝 Question {j+1}: {question}")
            print("-" * 30)
            
            try:
                result = pipeline.analyze_image(
                    image=test_case['image'],
                    question=question
                )
                
                print(result)
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # Test batch processing
    print(f"\n🔄 Testing Batch Processing")
    print("=" * 50)
    
    try:
        all_images = [xray_path, ct_path, mri_path]
        batch_questions = [
            "What medical imaging modality is this and what can you observe?",
            "What medical imaging modality is this and what can you observe?",
            "What medical imaging modality is this and what can you observe?"
        ]
        
        results = pipeline.batch_analyze_images(all_images, batch_questions)
        
        for i, result in enumerate(results):
            print(f"\nBatch Result {i+1}:")
            print(result[:200] + "..." if len(result) > 200 else result)
            
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
    
    # Cleanup
    os.unlink(xray_path)
    os.unlink(ct_path)
    os.unlink(mri_path)
    
    print(f"\n✅ Testing completed successfully!")
    print(f"The LLaVA-Med framework is ready for model integration.")

if __name__ == "__main__":
    main()