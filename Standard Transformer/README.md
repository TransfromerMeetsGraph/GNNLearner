# The Standard Transformer Model
The standard Transformer models are trained on the [Fairseq toolkit](https://github.com/pytorch/fairseq).

## Environment.
The Docker can be pulled from: apeterswu/fairseq_master:2021-5, `docker pull apeterswu/fairseq_master:2021-5`.

It builds the Pytorch (version == 1.6.0), Python (version == 3.7), and Fairseq (version == 0.10.2).

## Data Preprocess.
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

## Pretrained Models.
We provide our pretrained models.

 Model | # params | URL
 --- | --- | --- 
l1_loss_reg_comb_seed22.checkpoint119.pt| 83M | [model](https://mailustceducn-my.sharepoint.com/:u:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Standard%20Transformer/final_models/l1_loss_reg_comb_seed22.checkpoint119.pt?csf=1&web=1&e=Zaft3M)
l1_loss_reg_comb_seed4444.checkpoint119.pt| 83M | [model](https://mailustceducn-my.sharepoint.com/:u:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Standard%20Transformer/final_models/l1_loss_reg_comb_seed4444.checkpoint119.pt?csf=1&web=1&e=LLqF0k)
l1_loss_reg_comb_seed666666.checkpoint119.pt| 83M | [model](https://mailustceducn-my.sharepoint.com/:u:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Standard%20Transformer/final_models/l1_loss_reg_comb_seed666666.checkpoint119.pt?csf=1&web=1&e=J7HBfG)
l1_loss_reg_comb_seed88888888.checkpoint119.pt| 83M | [model](https://mailustceducn-my.sharepoint.com/:u:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Standard%20Transformer/final_models/l1_loss_reg_comb_seed88888888.checkpoint119.pt?csf=1&web=1&e=TbKWG3)

## Reproduce Inference. 
After downloading the pretrained models, put the models in a folder (e.g., ./checkpoints), you can reproduce our predictions (single model prediction and ensemble prediction):
```shell
bash reproduce.sh
```

## Train Your Own Models.
To train your own models, first preprocess the data as described above. Then train the model as follows:
```shell
SEED=${1:-22}
fairseq-train ./data/data_bin \
    --task sentence_prediction \
    --save-dir ./checkpoints/l1_loss_reg_comb_seed$SEED} \
    --arch roberta_base \
    --user-dir ./pylib/ext \
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
    --seed $SEED \
    --fp16
```

For inference, you can simply modify the `reproduce.sh` or do the single model inference, for example:
```
python infer_test.py $DATA ./checkpoints/l1_loss_reg_comb_seed22 checkpoint119.pt --bsz 64
```
