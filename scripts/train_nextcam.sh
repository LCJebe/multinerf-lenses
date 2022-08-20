#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

SCENE=grand_piano_2
EXPERIMENT=nextcam_scratch
DATA_DIR=/home/jebe/multinerf-lenses/data
CHECKPOINT_DIR=/home/jebe/multinerf-lenses/results/"$EXPERIMENT"/"$SCENE"

python -m train \
  --gin_configs=configs/nextcam.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}.npz'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr