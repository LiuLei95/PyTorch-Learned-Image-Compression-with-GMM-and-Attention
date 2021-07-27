#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python train.py --config examples/example/config_2048_256.json \
-n baseline_2048_256 --pretrain ./checkpoints/baseline_2048_256/iter_850000.pth.tar

