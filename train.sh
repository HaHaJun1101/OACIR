#!/bin/bash

# ==============================================================================
# Training Script for OACIRR and Standard CIR Datasets
# ==============================================================================

# -------------------------- Core Settings --------------------------
# Dataset to train on. Choices: ['Fashion', 'Car', 'Product', 'Landmark', 'Union', 'CIRR', 'FashionIQ']
DATASET="Union"

# Root directory of the dataset
DATA_ROOT="./Datasets/OACIRR"

# Model registry name. Choices: ['oacir_baseline', 'oacir_adafocal', 'oacir_adafocal_vector']
MODEL_NAME="oacir_adafocal"

# Path to the pre-trained model weight for initialization (OPTIONAL)
# Leave empty "" to use the default BLIP-2 pretrained Q-Former from LAVIS.
# Provide a path only if you want to resume training from a local checkpoint.
MODEL_WEIGHT=""

# ViT backbone. Choices: ['pretrain' (ViT-G), 'pretrain_vitL' (ViT-L)]
VIT_BACKBONE="pretrain"

# Directory to save checkpoints
SAVE_DIR="./checkpoints"


# ---------------------- Training Hyperparameters -------------------
EPOCHS=50
LR=1e-5
BATCH_SIZE=256
NUM_WORKERS=6
VAL_FREQ=1
LOSS_ALIGN=1.0
SEED=2026


# ------------------ Image Preprocessing Settings -------------------
# Transform type. Choices: ['squarepad', 'targetpad']
TRANSFORM="targetpad"
TARGET_RATIO=1.25


# -------------------- Visual Baseline Settings ---------------------
# To train the Visual Anchor Baseline (Drawing Bbox on image), set BBOX_WIDTH > 0 (e.g., 3)
BBOX_WIDTH=0
BBOX_COLOR="red"

# To train the ROI-Crop Baseline (Cropping the Bbox region), uncomment the line below:
# BBOX_CROP="--bounding-box-crop"


# -------------------------- Boolean Flags --------------------------
# AdaFocal mechanism (Keep uncommented if training AdaFocal)
HIGHLIGHT_TRAINING="--highlight-training"
HIGHLIGHT_INFERENCE="--highlight-inference"

# Save strategies
SAVE_TRAINING="--save-training"
SAVE_BEST="--save-best"

# Hardware / Memory options
SAVE_MEMORY="--save-memory"

# Text Prompt
# TEXT_ENTITY="--text-entity"


# ==============================================================================
# Execute Python Script
# ==============================================================================

echo "Starting Training for ${DATASET} using model ${MODEL_NAME}..."

python train.py \
    --dataset ${DATASET} \
    --data-root ${DATA_ROOT} \
    --blip-model-name ${MODEL_NAME} \
    --blip-model-weight ${MODEL_WEIGHT} \
    --vit-backbone ${VIT_BACKBONE} \
    --save-dir ${SAVE_DIR} \
    --num-epochs ${EPOCHS} \
    --learning-rate ${LR} \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --validation-frequency ${VAL_FREQ} \
    --loss-align ${LOSS_ALIGN} \
    --transform ${TRANSFORM} \
    --target-ratio ${TARGET_RATIO} \
    --bounding-box-width ${BBOX_WIDTH} \
    --bounding-box-color ${BBOX_COLOR} \
    --seed ${SEED} \
    ${BBOX_CROP} \
    ${HIGHLIGHT_TRAINING} \
    ${HIGHLIGHT_INFERENCE} \
    ${SAVE_TRAINING} \
    ${SAVE_BEST} \
    ${SAVE_MEMORY} \
    ${TEXT_ENTITY}