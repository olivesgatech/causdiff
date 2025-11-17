#!/usr/bin/env bash
OPTS="--model_dir=/home/hice1/skim3513/AIFirst_F24_data/darai/models/causdiff\
      --results_dir=/home/hice1/skim3513/AIFirst_F24_data/darai/models/causdiff \
      --mapping_file=./datasets/darai/mapping_l2_changed.txt \
      --mapping_coarse_file=./datasets/darai/mapping_l1_changed.txt \
      --vid_list_file_test=./datasets/darai/splits/test_split.txt \
      --vid_list_file=./datasets/darai/splits/train_split.txt \
      --gt_path=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/darai/groundTruth_img \
      --features_path=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/darai/features_img \
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
      --bz=8 \
      --lr=0.0005 \
      --model=bit-diff-pred-tcn \
      --num_epochs=105 \
      --epoch=80 \
      --num_stages=5 \
      --obs_stages=0 \
      --ant_stages=5 \
      --num_layers=9 \
      --channel_dropout_prob=0.4 \
      --num_highlevel_classes=512 \
      --date=202510290020 \
      --sample_rate=15"

python ./src/main-causal_hierarchical_onlylora_25b_darail2.py $OPTS


