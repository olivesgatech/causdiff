#!/usr/bin/env bash
OPTS="--model_dir=/mnt/data-tmp/seulgi/causdiff/model/darai/diff \
      --results_dir=/mnt/data-tmp/seulgi/causdiff/model/darai/diff \
      --mapping_file=./datasets/darai/mapping_l3_changed.txt \
      --mapping_coarse_file=./datasets/darai/mapping_l1_changed.txt \
      --vid_list_file_test=./datasets/darai/splits/test_split.txt \
      --vid_list_file=./datasets/darai/splits/train_split.txt \
      --gt_path=./datasets/darai/groundTruth_nov11/ \
      --features_path=./datasets/darai/features_temp \
      --split=4 \
      --conditioned_x0 \
      --use_features \
      --use_inp_ch_dropout \
      --layer_type gated \
      --part_obs \
      --num_diff_timesteps 1000 \
      --diff_loss_type l2 \
      --diff_obj pred_x0 \
      --action=train \
      --ds=bf \
      --bz=16 \
      --lr=0.0005 \
      --model=bit-diff-pred-tcn \
      --num_epochs=100 \
      --epoch=100 \
      --num_stages=5 \
      --obs_stages=0 \
      --ant_stages=5 \
      --num_layers=9 \
      --channel_dropout_prob=0.4 \
      --num_highlevel_classes=48 \
      --date=202508281716 \
      --sample_rate=3"

python ./src/main.py $OPTS


