# The Standard Transformer Model
The standard Transformer models are trained on the [Fairseq toolkit](https://github.com/pytorch/fairseq).

## Environment
The Docker can be pulled from: apeterswu/fairseq_master:2021-5, `docker pull apeterswu/fairseq_master:2021-5`.

It builds the Pytorch (version == 1.6.0), Python (version == 3.7), and Fairseq (version == 0.10.2).

## Data
We train the standard Transformer models on the official training + dev datasets. 
After download the raw tarining and dev data, preprocess the data (canonicalize and tokenize) as in [Two-branch Transformer](https://github.com/TransfromerMeetsGraph/GNNLearner/tree/dev/Two-branch%20Transformer#data-preprocessing), before binarizing. 
Note that following processes are in [Two-branch Transformer](https://github.com/TransfromerMeetsGraph/GNNLearner/tree/dev/Two-branch%20Transformer#data-preprocessing) folder.
```shell
# You may change the DATA and BINDIR by yourself.
DATA=./data/raw_data
BINDIR=./data/data_bin/input0
for subset in train valid test; do
    python molecule/canonicalize.py $DATA/${subset}.x --workers 30
done 
for subset in train valid test; do
    python molecule/tokenize_re.py $DATA/${subset}.x.can \
        --workers 30 --output-fn $DATA/${subset}.x.bpe 
done 
```
Then concatenate the training and dev data (also the label data):
```shell
cat $DATA/train.x.bpe $DATA/valid.x.bpe > $DATA/train.comb.x.bpe
cat $DATA/train.y $DATA/valid.y > $DATA/train.comb.y
```

Then binarize the source data as follow:
```shell
# Note that the provided dictionary (--srcdict) is counted on the training data.
fairseq-preprocess \
    --only-source \
    --trainpref "${DATA}/train.comb.x.bpe" \
    --validpref "${DATA}/valid.x.bpe" \
    --testpref "${DATA}/test.x.bpe" \
    --srcdict ./molecule/kddcup/dict.txt \
    --destdir "${BINDIR}/input0" \
    --workers 16

```
Put the target data (label) in the binarized data folder:
```shell
tgtdir=${BINDIR}/label
mkdir -p $tgtdir
cp $DATA/train.comb.y $tgtdir/train.label
cp $DATA/valid.y $tgtdir/valid.label
```

## Pretrained Models
We provide our pretrained models.

 Model | # params | URL
 --- | --- | --- 
l1_loss_reg_comb_seed22.checkpoint119.pt| 83M | [model]()
l1_loss_reg_comb_seed22.checkpoint119.pt| 83M | [model]()
l1_loss_reg_comb_seed22.checkpoint119.pt| 83M | [model]()
l1_loss_reg_comb_seed22.checkpoint119.pt| 83M | [model]()

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
