This repository contains our solution of two branch Transformer for the PCQM4M-LSC track of the OGB-LSC. The code is originally forked from [Fairseq](https://github.com/pytorch/fairseq).

# Requirements and Installation

This project has following requirements:

* [PyTorch](http://pytorch.org/) version == 1.8.0
* PyTorch Geometric version == 1.6.3
* RDKit version == 2020.09.5

The Dockerfile is [provided](./Dockerfile), and you can use it with `docker build -t pretrainmol .`. You can also pull an image from Dockerhub with `docker pull teslazhu/pretrainmol36:latest`, and start the docker container with `docker run -it pretrainmol36:latest bash`.

To install the code from source, run following commands in the docker container:

```shell
git clone https://github.com/TransfromerMeetsGraph/GNNLearner
cd GNNLearner/Two-branch\ Transformer/
pip install -e . 
```

# Getting Started

## Download our pre-trained model and Inference

Our models are released at [Onedrive/Two-branch Transformer](https://mailustceducn-my.sharepoint.com/:f:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Two-branch%20Transformer?csf=1&web=1&e=Xkw9dW).

 Description | # params | Dataset | Model
 --- | --- | --- | ---
Models trained on train + k-fold dev set|200M | [data](https://mailustceducn-my.sharepoint.com/:f:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Two-branch%20Transformer/dataset?csf=1&web=1&e=gvGHbL)/bindatadev(a/b/c/d) | [models](https://mailustceducn-my.sharepoint.com/:f:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Two-branch%20Transformer/k-fold?csf=1&web=1&e=fzy5vG)
Models trained on train + all dev set| 200M|[data](https://mailustceducn-my.sharepoint.com/:f:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Two-branch%20Transformer/dataset/bindatadev?csf=1&web=1&e=Phaf00)|[models](https://mailustceducn-my.sharepoint.com/:f:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Two-branch%20Transformer/all-dev?csf=1&web=1&e=0IVwsc)

By downloading the binarized data and pre-trained models, you can re-produce our scores with

```bash
git clone https://github.com/TransfromerMeetsGraph/GNNLearner
docker pull teslazhu/pretrainmo36:latest
docker run -it -v "${PWD}:/tmp/GNNLearner" teslazhu/pretrainmo36:latest bash

# Then run following commands in docker container
model=/path/to/model/checkpoint
data=/path/to/data

cd /tmp/GNNLearner/Two-branch\ Transformer
pip install -e .
python molecule/inference.py $model --dataset $data --subset valid/test
```

## Train with official dev set

### Data Preprocessing

See [DataPreprocessing](../DataPreprocessing/README.md) Section "Two-branch Transformer Data Processing Steps" for more details.

### Train

```shell
DATADIR=${BINDIR}
SAVEDIR=/yoursavedir
fairseq-train $DATADIR \
    --max-positions 512 --batch-size 64 \
    --task kddcup --required-batch-size-multiple 1 \
    --arch doublemodel --criterion graph_sp --dropout 0.1 \
    --attention-dropout 0.1 --weight-decay 0.01 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 --lr-scheduler polynomial_decay --lr 2e-4 \
    --total-num-update 297399 --warmup-updates 17844 --max-epoch 50 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method truncate --find-unused-parameters --update-freq 1 \
    --save-dir $SAVEDIR \
    --pooler-dropout 0 --relu-dropout 0 \
    --datatype tt --scaler-label --use-byol
```

### Inference

```shell
cktpath=/yourcktpath
DATADIR=${BINDIR}
groundtruthfn=/groundtruth_fn
python molecule/inference.py $cktpath --dataset $DATADIR \
    --label-fn $groundtruthfn
```

## Train with k-fold cross-validation

You can use [data_splitter.py](./data_splitter.py) to split the dev set, and use these data for 4-fold cross-validation. Preprocess and train your model again.

## Train with all dev set

You can cat all dev set into training set, and preprocess and train your model again.
