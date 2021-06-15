CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=/tmp/standard_transformer/pylib

MODEL=roberta
DATA=/tmp/standard_transformer/data
MODEL_FOLDER=/tmp/standard_transformer/checkpoints

mkdir -p predictions

TF_models=(
"l1_loss_reg_comb_seed22.checkpoint119.pt"
"l1_loss_reg_comb_seed4444.checkpoint119.pt"
"l1_loss_reg_comb_seed666666.checkpoint119.pt"
"l1_loss_reg_comb_seed88888888.checkpoint119.pt"
)

for modelname in ${TF_models[@]}; do
  python infer_test.py $DATA $MODEL_FOLDER $modelname --bsz 64
done

python average_transformer.py 
