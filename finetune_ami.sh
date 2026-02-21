#!/usr/bin/env bash

# "data" specifies the dataset name
# "train_path" specifies the training data path
# "val_path" specifies the valid data path
set -euo pipefail

stage=1
stop_stage=1000
data=ami
gpuid=0
data_root=data/dump
hf_data_root=data/hf_dump
noise_suffix=_noise_snr0

. ./path.sh
. ./local/parse_options.sh

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
    for data_set in ihm_train ihm_dev ihm_eval; do
        python local/strip_padding_pt_chunks.py \
        --pt_dir data/dump/$data_set$noise_suffix \
        --noisy_wavscp dump/raw/$data_set$noise_suffix/wav.scp \
        --clean_wavscp dump/raw/$data_set/wav.scp \
        --dry_run 0 \
        --recursive
    done
fi

#if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
#    for data_set in ihm_train_noise_snr0 ihm_dev_noise_snr0 ihm_eval_noise_snr0; do
#        python local/pt_chunks_to_hf_parquet.py \
#            --pt_dir $data_root/$data_set \
#            --out_dir $hf_data_root/$data_set \
#            --shard_size 2000 \
#            --float_dtype float16
#    done
#fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    python finetune/robust_ger_ami.py --data ${data} \
            --train_path $data_root/ihm_train_noise_snr0 \
            --val_path $data_root/ihm_dev_noise_snr0 
fi
