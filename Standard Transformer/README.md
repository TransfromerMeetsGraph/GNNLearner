# The Standard Transformer Model

The standard Transformer models are trained on the [Fairseq toolkit](https://github.com/pytorch/fairseq).

## Environment

The Docker can be pulled from: `apeterswu/fairseq_master:2021-5`, run `docker pull apeterswu/fairseq_master:2021-5`.

It builds the Pytorch (version == 1.6.0), Python (version == 3.7), and Fairseq (version == 0.10.2).

## Data Preprocessing

See [DataPreprocessing](../DataPreprocessing/README.md) Section "Standard Transformer Data Processing Steps" for more details.

## Pretrained Models

We provide our pretrained models.

 Model | # params | URL
 --- | --- | ---
l1_loss_reg_comb_seed22.checkpoint119.pt| 83M | [model](https://mailustceducn-my.sharepoint.com/:u:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Standard%20Transformer/final_models/l1_loss_reg_comb_seed22.checkpoint119.pt?csf=1&web=1&e=Zaft3M)
l1_loss_reg_comb_seed4444.checkpoint119.pt| 83M | [model](https://mailustceducn-my.sharepoint.com/:u:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Standard%20Transformer/final_models/l1_loss_reg_comb_seed4444.checkpoint119.pt?csf=1&web=1&e=LLqF0k)
l1_loss_reg_comb_seed666666.checkpoint119.pt| 83M | [model](https://mailustceducn-my.sharepoint.com/:u:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Standard%20Transformer/final_models/l1_loss_reg_comb_seed666666.checkpoint119.pt?csf=1&web=1&e=J7HBfG)
l1_loss_reg_comb_seed88888888.checkpoint119.pt| 83M | [model](https://mailustceducn-my.sharepoint.com/:u:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Standard%20Transformer/final_models/l1_loss_reg_comb_seed88888888.checkpoint119.pt?csf=1&web=1&e=TbKWG3)

## Reproduce Inference

After downloading the pretrained models, put the models in a folder (e.g., ./checkpoints), you can reproduce our predictions (single model prediction and ensemble prediction):

```bash
git clone https://github.com/TransfromerMeetsGraph/GNNLearner

mkdir -p ./GNNLearner/Standard\ Transformer/data
mv /path/to/processed/dataset/* ./GNNLearner/Standard\ Transformer/data
mkdir -p ./GNNLearner/Standard\ Transformer/checkpoints
mv /path/to/pretrained/checkpoints/*.pt ./GNNLearner/Standard\ Transformer/checkpoints

docker pull apeterswu/fairseq_master:2021-5
docker run -it -v "${PWD}/Standard Transformer:/tmp/standard_transformer" apeterswu/fairseq_master:2021-5 bash

# Then run following commands in docker container
cd /tmp/standard_transformer
bash reproduce.sh
```

## Train Your Own Models

To train your own models, first preprocess the data as described above. Then train the model as follows:

```bash
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

```bash
python infer_test.py $DATA ./checkpoints/l1_loss_reg_comb_seed22 checkpoint119.pt --bsz 64
```
