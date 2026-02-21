#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa

set -euo pipefail

stage=0
stop_stage=1000
generate_sets="ihm_train ihm_dev ihm_eval"
snr_affix=_noise_snr0
gpuid=0

. ./local/parse_options.sh
. ./path.sh

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    
    for gset in $generate_sets; do
        mkdir -p data/dump/${gset}$snr_affix
        CUDA_VISIBLE_DEVICES=$gpuid python local/generate_robust_hp_fast.py --resume \
            --noisy_wavscp dump/raw/${gset}${snr_affix}/wav.scp \
            --clean_wavscp dump/raw/${gset}/wav.scp\
            --text_path dump/raw/${gset}/text \
            --save_every 5000 \
            --output_path data/dump/${gset}${snr_affix}/${gset}.pt
    done
fi

