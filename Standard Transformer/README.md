# The Standard Transformer Model

## Reproduce inference 
to re-produce our results, please run `bash reproduce.sh`


The Transformer model setting:

data: train + all valid
data binarize script: /blob2/v-lijuwu/kdd/make_train_valid_bin.sh
model description: Transformer roberta setting with some modifications
code path: /blob2/v-lijuwu/kdd/code/ext (sentence_prediction_l1_loss.py, __init__.py)
docker image: retachet/ruamel:base
docker set up: pip install fairseq==0.10.2 pyarrow
training loss: l1 loss
training script: /blob2/v-lijuwu/kdd/code/run_scripts/train_regression_l1_final.sh
infer script: /blob2/v-lijuwu/kdd/code/infer_test.sh (it will call /blob2/v-lijuwu/kdd/code/infer.py)
infer_ensemble script: /blob2/v-lijuwu/kdd/code/ensem_test.sh (it will call /blob2/v-lijuwu/kdd/code/ensem_test.py, which will ensemble the results of predicted npz files)

detailed training setting and command:
fairseq-train /blob2/v-lijuwu/kdd/data/data_bin/reg_comb \
    --task sentence_prediction \
    --arch roberta_base \
    --user-dir $PREFIX/code/ext \
    --encoder-layers 12 \
    --encoder-embed-dim 768 \
    --encoder-ffn-embed-dim 3072 \
    --encoder-attention-heads 12 \
    --max-positions 512 \
    --criterion sentence_prediction_l1_criterion \
    --regression-target \
    --best-checkpoint-metric loss \
    --num-classes 1 \
    --batch-size 64 \
    --init-token 0 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --activation-dropout 0.0 \
    --pooler-dropout 0.0 \
    --weight-decay 0.01 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 0.0 \
    --lr 2e-4 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 12000 \
    --max-update 200000 \
    --update-freq 8 \
    --find-unused-parameters \
    --fp16
