#!/usr/bin/env bash
OPTS="--model_dir=/home/hice1/skim3513/AIFirst_F24_data/darai/models/causdiff\
      --results_dir=/home/hice1/skim3513/AIFirst_F24_data/darai/models/causdiff \
      --mapping_file=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/nturgbd/mapping_l2_changed.txt \
      --mapping_coarse_file=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/nturgbd/mapping_l2_changed.txt \
      --vid_list_file_test=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/nturgbd/splits/test_split_s001.txt \
      --vid_list_file=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/nturgbd/splits/train_split_s001.txt \
      --gt_path=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/nturgbd/groundTruth \
      --features_path=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/nturgbd/features \
      --split=4 \
      --conditioned_x0 \
      --use_features \
      --use_inp_ch_dropout \
      --layer_type gated \
      --part_obs \
      --num_diff_timesteps 1000 \
      --diff_loss_type l2 \
      --diff_obj pred_x0 \
      --action=val \
      --ds=bf \
      --bz=8 \
      --lr=0.0005 \
      --model=bit-diff-pred-tcn \
      --num_epochs=105 \
      --epoch=155 \
      --num_stages=5 \
      --obs_stages=0 \
      --ant_stages=5 \
      --num_layers=9 \
      --channel_dropout_prob=0.4 \
      --num_highlevel_classes=512 \
      --date=202511072137 \
      --sample_rate=1"

python ./src/main-causal_hierarchical_onlylora_7b_nturgbd.py $OPTS


