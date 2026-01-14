#!/bin/bash
set -e

# Default settings
OUTPUT_DIR="norwegian_dataset"
MODEL_SAVE_DIR="norwegian_model"
NB_TALE_TRANSCRIPTION="Annotation/part_1.trans"
NB_TALE_AUDIO_DIR="nb_tale_audio"

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Preparing dataset..."
# You can customize flags here, e.g. --limit 1000 for testing
# Ensure you have downloaded and extracted the NB Tale dataset if you want to use it.
# If NB Tale is not present, it will be skipped (if --nb-tale-audio-dir is missing or empty).
python prepare_dataset.py \
    --output-dir "$OUTPUT_DIR" \
    --nb-tale-transcription "$NB_TALE_TRANSCRIPTION" \
    --nb-tale-audio-dir "$NB_TALE_AUDIO_DIR"

echo "Starting training..."
python train.py \
    --input-dir "$OUTPUT_DIR" \
    --save-dir "$MODEL_SAVE_DIR"

echo "Training complete. Model saved in $MODEL_SAVE_DIR"
