#!/bin/bash

TRAINING_DATA=$1
OUTPUT_DIR=$2
EPOCHES=$3

mkdir -p $OUTPUT_DIR

module purge
module load python-env/2019.3
source ../env/bin/activate

mkdir -p $OUTPUT_DIR

python ../transformers/examples/run_language_modeling.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAINING_DATA \
    --mlm \
    --num_train_epochs $EPOCHES \
    --save_steps 10000 \
    --should_continue \
    --overwrite_output_dir

