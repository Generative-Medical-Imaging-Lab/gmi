#!/bin/bash

# Run all modular studies using the CLI (gmi train-image-reconstructor)
# This script loops over all combinations and calls the CLI for each

CONFIG_DIR="examples/modular_configs"
DATASETS=(mnist bloodmnist chestmnist)
SIMULATORS=(low_noise medium_noise high_noise)
# Define reconstructors for different channel counts
RECONSTRUCTORS_1CH=(simple_cnn_1ch linear_conv_1ch diffusers_unet_28_1ch)
RECONSTRUCTORS_3CH=(simple_cnn_3ch linear_conv_3ch diffusers_unet_28_3ch)

# Define which datasets use which channel count
# BloodMNIST: 3 channels (RGB), MNIST/ChestMNIST: 1 channel (grayscale)
declare -A DATASET_CHANNELS
DATASET_CHANNELS["mnist"]=1
DATASET_CHANNELS["bloodmnist"]=3
DATASET_CHANNELS["chestmnist"]=1

# Calculate total studies
TOTAL=0
for DATASET in "${DATASETS[@]}"; do
  CHANNELS=${DATASET_CHANNELS[$DATASET]}
  if [ "$CHANNELS" -eq 1 ]; then
    TOTAL=$((TOTAL + ${#SIMULATORS[@]} * ${#RECONSTRUCTORS_1CH[@]}))
  else
    TOTAL=$((TOTAL + ${#SIMULATORS[@]} * ${#RECONSTRUCTORS_3CH[@]}))
  fi
done

COUNT=0
SUCCESS_COUNT=0
FAILED_STUDIES=()

for DATASET in "${DATASETS[@]}"; do
  CHANNELS=${DATASET_CHANNELS[$DATASET]}
  
  if [ "$CHANNELS" -eq 1 ]; then
    RECONSTRUCTORS=("${RECONSTRUCTORS_1CH[@]}")
    echo "Using 1-channel reconstructors for $DATASET"
  else
    RECONSTRUCTORS=("${RECONSTRUCTORS_3CH[@]}")
    echo "Using 3-channel reconstructors for $DATASET"
  fi
  
  for SIMULATOR in "${SIMULATORS[@]}"; do
    for RECONSTRUCTOR in "${RECONSTRUCTORS[@]}"; do
      COUNT=$((COUNT+1))
      EXPERIMENT_NAME="${DATASET}_${SIMULATOR}_${RECONSTRUCTOR}"
      
      echo ""
      echo "============================================================"
      echo "Study $COUNT/$TOTAL: $DATASET + $SIMULATOR + $RECONSTRUCTOR"
      echo "Channels: $CHANNELS, Experiment: $EXPERIMENT_NAME"
      echo "============================================================"
      
      # Run the training command with error handling
      if gmi train-image-reconstructor \
        $CONFIG_DIR/training_config.yaml \
        --train-dataset $CONFIG_DIR/datasets/${DATASET}_train.yaml \
        --measurement-simulator $CONFIG_DIR/measurement_simulators/${SIMULATOR}.yaml \
        --image-reconstructor $CONFIG_DIR/image_reconstructors/${RECONSTRUCTOR}.yaml \
        --device cuda \
        --experiment-name "$EXPERIMENT_NAME"; then
        echo "✅ Study $COUNT completed successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT+1))
        
        # Note: WandB data is now automatically downloaded at the end of training
      else
        echo "❌ Study $COUNT failed: $DATASET + $SIMULATOR + $RECONSTRUCTOR"
        FAILED_STUDIES+=("$EXPERIMENT_NAME")
      fi
    done
  done
done

echo ""
echo "============================================================"
echo "All studies completed!"
echo "Successful: $SUCCESS_COUNT/$TOTAL"
echo "Failed: $((TOTAL - SUCCESS_COUNT))/$TOTAL"

if [ ${#FAILED_STUDIES[@]} -gt 0 ]; then
  echo ""
  echo "Failed studies:"
  for study in "${FAILED_STUDIES[@]}"; do
    echo "  - $study"
  done
fi
echo "============================================================" 