#!/bin/bash

# Run image reconstruction training from YAML config
# This script runs the training command inside the Docker container

echo "🚀 Starting image reconstruction training..."
echo "Using config: config.yaml"

# Get the absolute path to the config file
SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
CONFIG_PATH="$SCRIPT_DIR/config.yaml"

# Change to project root directory (two levels up from examples/medmnist_restoration_from_config)
cd "$SCRIPT_DIR/../.."

# Run the training command using the gmi CLI with absolute path
gmi train-image-reconstructor "$CONFIG_PATH" --device cuda

echo "✅ Training completed!" 