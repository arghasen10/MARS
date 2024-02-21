#!/bin/bash
#SBATCH --gres=gpu:1

cd 3d_interpolation && srun sh tf_interpolate_compile.sh
cd ../grouping && srun sh tf_grouping_compile.sh
cd ../sampling && srun sh tf_sampling_compile.sh
