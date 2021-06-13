#!/bin/bash
#
# The model inference script.
#
# Usage: bash ./scripts/train/inference.sh [arch1|arch2|arch3|arch4|arch5|:all] [valid]

ProjectDir=/workspace/exp-ogb # [NOTE]: You can change it into your root path.
PythonVersion=$(python -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')
PyTorchVersion=$(python -c 'import torch; print("{}.{}".format(*torch.__version__.split(".")))')
CudaVersion=$(python -c 'import torch; print("cu{}{}".format(*torch.version.cuda.split(".")[:2]))')

LogDir=${ProjectDir}/log
CheckpointDir=${ProjectDir}/checkpoints
SavedTestDir=${ProjectDir}/saved_tests

mkdir -p ${SavedTestDir}

InferCheckpoint=${1}
if [[ "x${InferCheckpoint}" == "x" ]]; then
    InferCheckpoint=arch1
fi
InferSubset=${2}
if [[ "x${InferSubset}" == "x" ]]; then
    InferSubset=test
fi

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
[[ "${InferCheckpoint}" == "arch1" || "${InferCheckpoint}" == ":all" ]] && {
    python ${ProjectDir}/examples/lsc/pcqm4m/test_inference_gnn.py \
        --dataset_dir ${ProjectDir}/dataset \
        --gnn gin-virtual --graph_pooling mean \
        --num_layers 5 --batch_size 256 \
        --drop_ratio 0.1 --emb_dim 1024 --residual False \
        --checkpoint_dir ${CheckpointDir}/arch1 --save_test_dir ${SavedTestDir}/arch1 \
        --eval-subsets ${InferSubset}
}

# GNN setting 2
[[ "${InferCheckpoint}" == "arch2" || "${InferCheckpoint}" == ":all" ]] && {
    python ${ProjectDir}/examples/lsc/pcqm4m/test_inference_gnn.py \
        --dataset_dir ${ProjectDir}/dataset \
        --gnn gin-virtual --graph_pooling mean \
        --num_layers 5 --batch_size 256 \
        --drop_ratio 0.1 --emb_dim 1024 --residual False \
        --seed 42 \
        --checkpoint_dir ${CheckpointDir}/arch2 --save_test_dir ${SavedTestDir}/arch2 \
        --eval-subsets ${InferSubset}
}

# GNN setting 3
[[ "${InferCheckpoint}" == "arch3" || "${InferCheckpoint}" == ":all" ]] && {
    python ${ProjectDir}/examples/lsc/pcqm4m/test_inference_gnn.py \
        --dataset_dir ${ProjectDir}/dataset \
        --gnn gin-virtual --graph_pooling mean \
        --num_layers 5 --batch_size 256 \
        --drop_ratio 0.1 --emb_dim 1024 --residual False \
        --checkpoint_dir ${CheckpointDir}/arch3 --save_test_dir ${SavedTestDir}/arch3 \
        --eval-subsets ${InferSubset}
}

# GNN setting 4
[[ "${InferCheckpoint}" == "arch4" || "${InferCheckpoint}" == ":all" ]] && {
    python ${ProjectDir}/examples/lsc/pcqm4m/test_inference_gnn.py \
        --dataset_dir ${ProjectDir}/dataset \
        --gnn gin-virtual --graph_pooling mean \
        --num_layers 5 --batch_size 256 \
        --drop_ratio 0.1 --emb_dim 1024 --residual False \
        --checkpoint_dir ${CheckpointDir}/arch4 --save_test_dir ${SavedTestDir}/arch4 \
        --eval-subsets ${InferSubset}
}

# GNN setting 5
[[ "${InferCheckpoint}" == "arch5" || "${InferCheckpoint}" == ":all" ]] && {
    python ${ProjectDir}/examples/lsc/pcqm4m/test_inference_gnn.py \
        --dataset_dir ${ProjectDir}/dataset \
        --gnn gin-virtual --graph_pooling mean \
        --num_layers 5 --batch_size 512 \
        --drop_ratio 0.1 --emb_dim 1024 --residual True \
        --seed 7 \
        --checkpoint_dir ${CheckpointDir}/arch5 --save_test_dir ${SavedTestDir}/arch5 \
        --eval-subsets ${InferSubset}
}
