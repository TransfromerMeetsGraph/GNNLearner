# GNNLearner

## Introduction
In this repo, we provide our solution to KDD cup 2021. We use three types of models:

- GIN network, which is the GIN with virtual nodes. The code is in the folder `GIN`; 
- Standard Transformer, which is the same as that in NLP. The code is in the folder `Standard Transformer`;
- Two-branch Transformer, which is a variant of Transformer with a regression branch and a classification branch. The code is in the folder `Two-branch Transformer`.

Please refer to the README in each folder for more details.

## Pre-trained checkpoints and predictions
Our pre-trained models are released on [Onedrive](https://mailustceducn-my.sharepoint.com/:f:/g/personal/teslazhu_mail_ustc_edu_cn/Emvau85Qcv9NowWIrR3d5HEBidQuh8sH9yOcgoAdqdd-Ag?e=YgHdN9). 

Our all prediction are released at [final_v0](https://mailustceducn-my.sharepoint.com/:f:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/final_v0?csf=1&web=1&e=WjNhqX). 

If you want to reproduce our results, go each subdirectory and follow the instructions to reconstruct the results. After that,  by running [submitted_predictions.py](./submitted_predictions.py), you can merge all predictions of each type of model.


## Data processing
The data processing steps are shown in [DataPreprocessing](./DataPreprocessing/README.md) folder.

## Paper
For technique details, see this [paper](./paper.pdf).
