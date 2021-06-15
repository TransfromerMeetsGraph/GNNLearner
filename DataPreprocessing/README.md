# Data Processing Steps

This file introduces data processing steps for `Standard Transformer` and `Two-branch Transformer` models. `GIN` models use the official data processing step.

This file is provided to researchers who want to reproduce the model training results from scratch. For inference, you can use the pre-built [datasets](https://mailustceducn-my.sharepoint.com/personal/teslazhu_mail_ustc_edu_cn/_layouts/15/onedrive.aspx?originalPath=aHR0cHM6Ly9tYWlsdXN0Y2VkdWNuLW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL3Rlc2xhemh1X21haWxfdXN0Y19lZHVfY24vRW12YXU4NVFjdjlOb3dXSXJSM2Q1SEVCaWRRdWg4c0g5eU9jZ29BZHFkZC1BZz9ydGltZT13MnQ5ZnZ3djJVZw&id=%2Fpersonal%2Fteslazhu%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2Fshare%2Fpublic%2Fkddcup%2FTwo%2Dbranch%20Transformer%2Fdataset) directly.

## Two-branch Transformer Data Processing Steps

1. Extract the raw SMILES-Score dataset from the official dataset

    ```bash
    pip install ogb
    ROOT=/path/to/GNNLearner
    cd ${ROOT}
    python DataPreprocessing/obtain_data.py ./data/raw
    ```

    After extraction, SMILES-Score dataset files are in `./data/SMILES`, `*.x` are SMILES strings, `*.y` are floating point HOMO-LUMO score labels.

2. Build binary dataset from raw dataset

    Assuming that you have placed your data (molecule SMILES and the corresponding label files) in one directory named DATADIR, we need to canonicalize, tokenize and binarize these data.

    ```bash
    # You should change the DATADIR and BINDIR for yourself.
    
    DATADIR=${ROOT}/data/raw
    BINDIR=${ROOT}/data/bindata

    cd ${ROOT}/Two-branch\ Transformer
    pip install -e .

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

## Standard Transformer Data Processing Steps

We train the standard Transformer models on the official training + dev datasets.
After download the raw tarining and dev data, preprocess the data (canonicalize and tokenize) as in Two-branch Transformer, before binarizing.
Note that following processes are in Two-branch Transformer (see the previous section).

```bash
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

```bash
cat $DATA/train.x.bpe $DATA/valid.x.bpe > $DATA/train.comb.x.bpe
cat $DATA/train.y $DATA/valid.y > $DATA/train.comb.y
```

Then binarize the source data as follow:

```bash
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

```bash
tgtdir=${BINDIR}/label
mkdir -p $tgtdir
cp $DATA/train.comb.y $tgtdir/train.label
cp $DATA/valid.y $tgtdir/valid.label
```
