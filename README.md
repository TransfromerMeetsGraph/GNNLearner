# GNNLearner

Solution of KDD cup 2021

We provided code in three sub-directories: GIN containing code on GIN networks, standard Transformer containing code on standard Transformer networks, and Two-branch Transformer containing code on two-branch Transformer. Our last submission is ensembled on these three networks. Please see corresponding README in each folder for more details.

Our pre-trained models are released on [Onedrive](https://mailustceducn-my.sharepoint.com/:f:/g/personal/teslazhu_mail_ustc_edu_cn/Emvau85Qcv9NowWIrR3d5HEBidQuh8sH9yOcgoAdqdd-Ag?e=YgHdN9). Our all prediction are released at [final_v0](https://mailustceducn-my.sharepoint.com/:f:/r/personal/teslazhu_mail_ustc_edu_cn/Documents/share/public/kddcup/final_v0?csf=1&web=1&e=WjNhqX). If you want to reproduce our results, go each subdirectory and follow the instructions. Last, by running [average.py](./average.py), you can merge all prediction.

Some data processing steps are shown in [DataProcessing](./DataProcessing/README.md) folder.

For technique details, see this [paper](./paper.pdf).
