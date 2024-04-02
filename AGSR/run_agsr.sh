#!/bin/bash

# Capture command-line arguments
EPOCHS=$1
LR=$2
STEP_SIZE=$3
GAMMA=$4

# Construct the log filename
LOG_FILENAME="agsr_${EPOCHS}_${LR}_${STEP_SIZE}_${GAMMA}_log.txt"

# Run the Python script with nohup
nohup python -u agsr_run.py --epochs $EPOCHS --lr $LR --step_size $STEP_SIZE --gamma $GAMMA > $LOG_FILENAME 2>&1 &