
#!/usr/bin/env bash
OPTS="--model_dir=/home/hice1/skim3513/scratch/GTDA/model/bf/determ \
      --results_dir=/home/hice1/skim3513/scratch/GTDA/model/bf/determ\
      --mapping_file=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/breakfast/mapping_l2.txt \
      --mapping_coarse_file=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/breakfast/mapping_l2.txt \
      --gt_path=./datasets/breakfast/groundTruth/ \
      --features_path=/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/breakfast/features \
      --vid_list_file_test=./datasets/breakfast/splits/test.split1.bundle \
      --vid_list_file=./datasets/breakfast/splits/test.split1.bundle \
      --split=1 \
      --ds=bf \
      --action=val \
      --use_features \
      --use_inp_ch_dropout \
      --part_obs \
      --layer_type gated \
      --bz=1 \
      --lr=0.0005 \
      --model=pred-tcn \
      --num_epochs=100 \
      --epoch=60 \
      --num_stages=5 \
      --obs_stages=0 \
      --ant_stages=5 \
      --num_layers=9 \
      --channel_dropout_prob 0.4 \
      --num_highlevel_classes=10 \
      --sample_rate=3"

python ./src/main.py $OPTS



