This repository contains our solution of two branch Transformer for the PCQM4M-LSC track of the OGB-LSC. The code is originally forked from [Fairseq](https://github.com/pytorch/fairseq).

# Requirements and Installation
The Dockerfile is [provided](./Dockerfile), and you can use it with `docker build -t pretrainmol .`.
* [PyTorch](http://pytorch.org/) version == 1.8.0
* PyTorch Geometric version == 1.6.3
* RDKit version == 2020.09.5

To install the code from source
```shell
git clone https://github.com/TransfromerMeetsGraph/GNNLearner
cd GNNLearner/Two-branch\ Transformer/
pip install -e . 
```
# Getting Started
## Train with official dev set
### Data Preprocessing 
Assuming that you have placed your data (molecule SMILES and the corresponding label files) in one directory named DATADIR, we need to canonicalize, tokenize and binarize these data.
```shell
# You should change the DATADIR and BINDIR for yourself.
DATADIR=/blob/v-jinhzh/data/kddcup/raw
BINDIR=/blob/v-jinhzh/data/kddcup/bindata
for subset in train valid test; do
    python molecule/canonicalize.py $DATADIR/${subset}.x --workers 30
done 
for subset in train valid test; do
    python molecule/tokenize_re.py $DATADIR/${subset}.x.can \
        --workers 30 --output-fn $DATADIR/${subset}.x.bpe 
done 


# Binarize the SMILES files
tgtdir=${BINDIR}/input0
mkdir -p $tgtdir

# We provide the dictionary we used for easy reproduction, and you can build it by yourself without the `--srcdict` option.
fairseq-preprocess \
    --only-source \
    --trainpref $DATADIR/train.x.bpe \
    --validpref $DATADIR/valid.x.bpe \
    --testpref $DATADIR/test.x.bpe \
    --destdir $tgtdir \
    --workers 30 \
    --molecule \
    --srcdict ./molecule/kddcup/dict.txt


# Convert the regression labels to classification labels
python check_label.py --input $DATADIR/train.y --dict ./molecule/kddcup/dict.17144.txt
python check_label.py --input $DATADIR/valid.y --dict ./molecule/kddcup/dict.17144.txt
# Binarize the classfication 
numel=17144
tgtdir=${BINDIR}/label_cls_${numel}
fairseq-preprocess \
    --only-source  \
    --trainpref $DATADIR/train.y.cls \
    --validpref $DATADIR/valid.y.cls \
    --destdir $tgtdir \
    --workers 30 \
    --srcdict ./molecule/kddcup/dict.17144.txt

tgtdir=${BINDIR}/label_reg
mkdir -p $tgtdir
for subset in train valid; do 
    cp $DATADIR/${subset}.y $tgtdir/${subset}.y
done

``` 
### Train 
```shell
DATADIR=${BINDIR}
SAVEDIR=\yoursavedir
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
### inference 
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


## Our pre-trained model 
Our models are released at [Onedrive](https://mailustceducn-my.sharepoint.com/:f:/g/personal/teslazhu_mail_ustc_edu_cn/Emvau85Qcv9NowWIrR3d5HEBidQuh8sH9yOcgoAdqdd-Ag?e=YgHdN9)/Two-branch Transformer.

 Description | # params | Dataset | Model 
 --- | --- | --- | ---
Models trained on k-fold dev set|200M | [data](https://mailustceducn-my.sharepoint.com/:f:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Two-branch%20Transformer/dataset?csf=1&web=1&e=gvGHbL)/bindatadev(a/b/c/d) | [models](https://mailustceducn-my.sharepoint.com/:f:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Two-branch%20Transformer/k-fold?csf=1&web=1&e=fzy5vG)
Models trained on all dev set| 200M|[data](https://mailustceducn-my.sharepoint.com/:f:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Two-branch%20Transformer/dataset/bindatadev?csf=1&web=1&e=Phaf00)|[models](https://mailustceducn-my.sharepoint.com/:f:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/Two-branch%20Transformer/all-dev?csf=1&web=1&e=0IVwsc)


By downloading the binarized data and pre-trained models, you can re-produce our scores with 
```shell 
python molecule/inerence.py $model --dataset $data
```



