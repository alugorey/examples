#!/bin/bash

# MI300 Envs
export HSA_DISABLE_CACHE=1
#export MIOPEN_DEBUG_CONV_WINOGRAD=0
#export MIOPEN_DEBUG_CONV_FFT=0
#export MIOPEN_DEBUG_CONV_DIRECT=1
#export MIOPEN_DEBUG_CONV_GEMM=0
#export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0

# F8 specific envs
export F8_CONFIRM=0
export TORCH_ALLOW_F8_ROCM_OVERRIDE_GEMM=0
export TORCH_ALLOW_F8_ROCM_OVERRIDE_CONV=1


# Limit GPUs
export HIP_VISIBLE_DEVICES=0


# rocBLAS output
#export ROCBLAS_LAYER=2

# rocBLAS numerical checking
#export ROCBLAS_CHECK_NUMERICS=2

#MIOpen numerical checking
#export MIOPEN_CHECK_NUMERICS=2

#MIOpen logging
#export MIOPEN_ENABLE_LOGGING=1
#export MIOPEN_ENABLE_LOGGING_MPMT=1
#export MIOPEN_ENABLE_LOGGING_CMD=1
#export MIOPEN_LOG_LEVEL=6


#python3 main.py -a resnet50 --lr 9.1 --wd 0.0002 --epochs 1 --batch-size 256 -j 32 -p 10 --gpu 0 /home/imagenet

python3 main.py -a resnet50 --print-freq 1 --epochs 1 --gpu 0 /home/imagenet

echo "Done!"
