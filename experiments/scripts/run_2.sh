#!/bin/bash

unset LD_PRELOAD

export PATH=/home/miproj/4thyr.oct2019/al826/anaconda3/bin:$PATH
source activate torch1.7

CUDAPATH="/usr/local/cuda-10.0/lib64:/home/mifs/ar527/bin/tensorflow-gpu/cudnn-8.0-linux-x64-v6.0/cuda/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64/"
export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE

python /home/alta/Conversational/OET/al826/2022/seq_cls/run_train.py --exp_name exp_9 --lr 1e-5 --bsz 4 --epochs 25 

# qsub -cwd -j yes -o 'LOGs/run_1.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' run.sh

