#!/bin/bash

# Visualize All Datasets Script
# This script visualizes all available datasets in the GMI package
# including MNIST and all MedMNIST variants with different sizes and splits
# 
# Usage: Run this script from the host, it will execute inside the Docker container

set -e  # Exit on any error

echo "🎯 Starting visualization of ALL datasets..."
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to visualize a dataset (runs directly in container)
visualize_dataset() {
    local dataset_name=$1
    print_status "Visualizing dataset: $dataset_name"
    
    if /usr/local/bin/gmi visualize-dataset --dataset "$dataset_name"; then
        print_success "✅ Completed visualization for: $dataset_name"
    else
        print_error "❌ Failed to visualize: $dataset_name"
        return 1
    fi
    echo ""
}

# Start with MNIST
print_status "Starting with MNIST dataset..."
visualize_dataset "mnist"

# MedMNIST variants
MEDMNIST_VARIANTS=(
    "PathMNIST"
    "ChestMNIST" 
    "DermaMNIST"
    "OCTMNIST"
    "PneumoniaMNIST"
    "RetinaMNIST"
    "BreastMNIST"
    "BloodMNIST"
    "TissueMNIST"
    "OrganAMNIST"
    "OrganCMNIST"
    "OrganSMNIST"
)

# Sizes and splits for MedMNIST
SIZES=(28 64 128 224)
SPLITS=("train" "val" "test")

# Counter for tracking progress
total_datasets=$((1 + ${#MEDMNIST_VARIANTS[@]} * ${#SIZES[@]} * ${#SPLITS[@]}))
current=0

print_status "Total datasets to visualize: $total_datasets"
echo "============================================================"

# Visualize all MedMNIST variants with all size/split combinations
# Loop through sizes first, then variants, then splits for better efficiency
for size in "${SIZES[@]}"; do
    print_status "Processing size: ${size}x${size}"
    
    for variant in "${MEDMNIST_VARIANTS[@]}"; do
        for split in "${SPLITS[@]}"; do
            current=$((current + 1))
            dataset_name="${variant}_${size}_${split}"
            
            print_status "Progress: $current/$total_datasets - $dataset_name"
            
            if visualize_dataset "$dataset_name"; then
                print_success "✅ $dataset_name completed successfully"
            else
                print_warning "⚠️  $dataset_name failed, continuing with next dataset..."
            fi
            
            # Small delay to avoid overwhelming the system
            sleep 1
        done
    done
done

echo "============================================================"
print_success "🎉 All dataset visualizations completed!"
print_status "📁 Visualizations saved to: gmi_data/outputs/visualizations/ (inside container)"
print_status "📊 Total datasets processed: $current"

# List all generated files (from container perspective)
echo ""
print_status "Generated visualization files:"
ls -la gmi_data/outputs/visualizations/ || print_warning "Could not list visualization files"

echo ""
print_success "✨ Script completed successfully!"
print_status "💡 To copy files from container to host, use:"
print_status "   docker cp gmi-container:/gmi_base/gmi_data/outputs/visualizations/ ./visualizations/" 