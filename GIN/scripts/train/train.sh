#!/bin/bash
#
# The model training script.

ProjectDir=/workspace/exp-ogb   # [NOTE]: You can change it into your root path.
PythonVersion=$(python -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')
PyTorchVersion=$(python -c 'import torch; print("{}.{}".format(*torch.__version__.split(".")))')
CudaVersion=$(python -c 'import torch; print("cu{}{}".format(*torch.version.cuda.split(".")[:2]))')

LogDir=${ProjectDir}/log
CheckpointDir=${ProjectDir}/checkpoints
SavedTestDir=${ProjectDir}/saved_tests

mkdir -p ${LogDir} ${CheckpointDir} ${SavedTestDir}

echo "
Python Version: ${PythonVersion}
PyTorch Version: ${PyTorchVersion}
Cuda Version: ${CudaVersion}
"

# Setup paths.
export PYTHONPATH="$(pwd)/.local/lib/python${PythonVersion}/site-packages:${PYTHONPATH}"
export PATH="$(pwd)/.local/bin:${PATH}"

# Installing packages.
echo "Installing OGB ..."
python setup.py develop --prefix .local

# GNN setting 1
python ${ProjectDir}/examples/lsc/pcqm4m/main_gnn.py \
    --dataset_dir ${ProjectDir}/dataset \
    --gnn gin-virtual --graph_pooling mean \
    --num_layers 5 --batch_size 256 \
    --drop_ratio 0.1 --emb_dim 1024 --residual False \
    --lr-scheduler step:s=60,g=0.1 \
    --log_dir ${LogDir}/arch1 --checkpoint_dir ${CheckpointDir}/arch1 --save_test_dir ${SavedTestDir}/arch1

## GNN setting 2
#python ${ProjectDir}/examples/lsc/pcqm4m/main_gnn.py \
#    --dataset_dir ${ProjectDir}/dataset \
#    --gnn gin-virtual --graph_pooling mean \
#    --num_layers 5 --batch_size 256 \
#    --drop_ratio 0.1 --emb_dim 1024 --residual False \
#    --seed 42 \
#    --log_dir ${LogDir}/arch2 --checkpoint_dir ${CheckpointDir}/arch2 --save_test_dir ${SavedTestDir}/arch2

## GNN setting 3
#python ${ProjectDir}/examples/lsc/pcqm4m/main_gnn.py \
#    --dataset_dir ${ProjectDir}/dataset \
#    --gnn gin-virtual --graph_pooling mean \
#    --num_layers 5 --batch_size 256 \
#    --drop_ratio 0.1 --emb_dim 1024 --residual False \
#    --log_dir ${LogDir}/arch3 --checkpoint_dir ${CheckpointDir}/arch3 --save_test_dir ${SavedTestDir}/arch3

## GNN setting 4
#python ${ProjectDir}/examples/lsc/pcqm4m/main_gnn.py \
#    --dataset_dir ${ProjectDir}/dataset \
#    --gnn gin-virtual --graph_pooling mean \
#    --num_layers 5 --batch_size 256 \
#    --drop_ratio 0.1 --emb_dim 1024 --residual False \
#    --lr-scheduler step:s=30,g=0.5 \
#    --log_dir ${LogDir}/arch4 --checkpoint_dir ${CheckpointDir}/arch4 --save_test_dir ${SavedTestDir}/arch4

## GNN setting 5
#python ${ProjectDir}/examples/lsc/pcqm4m/main_gnn.py \
#    --dataset_dir ${ProjectDir}/dataset \
#    --gnn gin-virtual --graph_pooling mean \
#    --num_layers 5 --batch_size 512 \
#    --drop_ratio 0.1 --emb_dim 1024 --residual True \
#    --seed 7 \
#    --log_dir ${LogDir}/arch5 --checkpoint_dir ${CheckpointDir}/arch5 --save_test_dir ${SavedTestDir}/arch5
