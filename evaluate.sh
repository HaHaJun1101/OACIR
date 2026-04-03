#!/bin/bash

# ==============================================================================
# Evaluation Script for OACIRR and Standard CIR Datasets
# ==============================================================================

# -------------------------- Core Settings --------------------------
# Dataset to evaluate on. Choices: ['Fashion', 'Car', 'Product', 'Landmark', 'CIRR', 'FashionIQ']
DATASET="Fashion"

# Root directory of the dataset
DATA_ROOT="./Datasets/OACIRR"

# Model registry name. Choices: ['oacir_baseline', 'oacir_adafocal', 'oacir_adafocal_vector']
MODEL_NAME="oacir_adafocal"

# Path to the fine-tuned model weight to be evaluated (REQUIRED)
# e.g., "./checkpoints/OACIR_Union_finetune_.../saved_models/adafocal_finetune_best.pt"
MODEL_WEIGHT="/path/to/your/finetuned/weight.pt"

# ViT backbone. Choices: ['pretrain' (ViT-G), 'pretrain_vitL' (ViT-L)]
VIT_BACKBONE="pretrain"


# ------------------ Image Preprocessing Settings -------------------
# Transform type. Choices: ['squarepad', 'targetpad']
TRANSFORM="targetpad"
TARGET_RATIO=1.25


# -------------------- Visual Baseline Settings ---------------------
# To evaluate the Visual Anchor Baseline (Drawing Bbox on image), set BBOX_WIDTH > 0 (e.g., 3)
BBOX_WIDTH=0
BBOX_COLOR="red"

# To evaluate the ROI-Crop Baseline (Cropping the Bbox region), uncomment the line below:
# BBOX_CROP="--bounding-box-crop"


# -------------------------- Boolean Flags --------------------------
# AdaFocal mechanism (Keep uncommented if evaluating AdaFocal)
HIGHLIGHT_INFERENCE="--highlight-inference"

# Save the detailed retrieval results as JSON for visualization
# SAVE_RESULTS="--save-results"

# Hardware/Memory options
SAVE_MEMORY="--save-memory"

# Text Prompt
# TEXT_ENTITY="--text-entity"


# ==============================================================================
# Execute Python Script
# ==============================================================================

echo "Starting Evaluation on ${DATASET} using model ${MODEL_NAME}..."

python evaluate.py \
    --dataset ${DATASET} \
    --data-root ${DATA_ROOT} \
    --blip-model-name ${MODEL_NAME} \
    --blip-model-weight ${MODEL_WEIGHT} \
    --vit-backbone ${VIT_BACKBONE} \
    --transform ${TRANSFORM} \
    --target-ratio ${TARGET_RATIO} \
    --bounding-box-width ${BBOX_WIDTH} \
    --bounding-box-color ${BBOX_COLOR} \
    ${BBOX_CROP} \
    ${HIGHLIGHT_INFERENCE} \
    ${SAVE_RESULTS} \
    ${SAVE_MEMORY} \
    ${TEXT_ENTITY}