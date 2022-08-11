#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

SCENE=portrait_flower
EXPERIMENT=nextcam
DATA_DIR=/home/jebe/multinerf/data
CHECKPOINT_DIR=/home/jebe/multinerf/results/"$EXPERIMENT"/"$SCENE"

rm "$CHECKPOINT_DIR"/*
python -m train \
  --gin_configs=configs/nextcam.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}.npz'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr
