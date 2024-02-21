#!/bin/sh
#cd ./tf_ops
#sh compile-ops.sh
#cd ..
module load anaconda3/5.1.0-gpu
srun -t 08:00:00 --gres=gpu:1 python train.py --max_epoch=40
srun -t 08:00:00 --gres=gpu:1 python train.py --max_epoch=200 --num_point=510
srun -t 08:00:00 --gres=gpu:1 python train.py --max_epoch=200 --num_point=5279 --num_frame=16

srun -t 24:00:00 --gres=gpu:1 --mem-per-cpu=80000 python train.py --max_epoch=400 --num_point=51 --num_frame=16 --model=gesturenet_sparse
