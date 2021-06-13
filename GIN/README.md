# TransformerMeetsGraph GNN Models

This repo is a fork of [official code](https://github.com/snap-stanford/ogb).
To get help from the official code, see the [original README](./README-original.md).

## Environment (Docker)

1. Requirements: CUDA=10.1, CuDNN=7, docker
2. Build docker:
   1. (Recommended) Pull from DockerHub: `docker pull fyabc/ogb-lsc-kdd21:main`;
   2. Or use the Dockerfile at `./scripts/train/Dockerfile`.

## Download Pre-trained Models and Inference

You can download our pre-trained models.

1. Download pre-trained Models and Predictions at [here](https://mailustceducn-my.sharepoint.com/personal/teslazhu_mail_ustc_edu_cn/_layouts/15/onedrive.aspx?originalPath=aHR0cHM6Ly9tYWlsdXN0Y2VkdWNuLW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL3Rlc2xhemh1X21haWxfdXN0Y19lZHVfY24vRW12YXU4NVFjdjlOb3dXSXJSM2Q1SEVCaWRRdWg4c0g5eU9jZ29BZHFkZC1BZz9ydGltZT1YRDZhS0UwdTJVZw&id=%2Fpersonal%2Fteslazhu%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2Fshare%2Fpublic%2Fkddcup%2FGIN),
put them into `/path/to/this/project/checkpoints`. The folder structure should be:

    ```text
    /path/to/this/project/
        checkpoints/
            arch1/
                checkpoint.pt
            arch2/
                checkpoint.pt
            ...
            predictions/
                ...
    ```

2. Inference checkpoints and ensemble:

    ```bash
    cd /path/to/this/project 
    docker run -v ${PWD}:/workspace/exp-ogb fyabc/ogb-lsc-kdd21:main bash -c "cd /workspace/exp-ogb; bash scripts/train/inference.sh :all"
    docker run -v ${PWD}:/workspace/exp-ogb fyabc/ogb-lsc-kdd21:main bash -c "cd /workspace/exp-ogb; \
    python scripts/train/evaluate.py -p saved_tests/arch[1-5]/y_pred_pcqm4m.npz -o saved_tests/ensemble_y_pred_pcqm4m.npz"
    ```

Then the output **saved_test/ensemble_y_pred_pcqm4m.npz** is the ensemble output prediction.

## Model Training

1. The model training script: [here](./scripts/train/train.sh)
2. Run the training and inference script in the docker:

    ```bash
    cd /path/to/this/project 
    docker run -v ${PWD}:/workspace/exp-ogb fyabc/ogb-lsc-kdd21:main bash -c "cd /workspace/exp-ogb; bash scripts/train/train.sh"
    docker run -v ${PWD}:/workspace/exp-ogb fyabc/ogb-lsc-kdd21:main bash -c "cd /workspace/exp-ogb; bash scripts/train/inference.sh"
    ```

3. If you want to train other settings, change the "GNN setting 1" in `train.sh` and `inference.sh` into other GNN settings.
4. Get ensemble predictions:

    ```bash
    docker run -v ${PWD}:/workspace/exp-ogb fyabc/ogb-lsc-kdd21:main bash -c "cd /workspace/exp-ogb; \
    python scripts/train/evaluate.py -p saved_tests/arch[1-5]/y_pred_pcqm4m.npz -o saved_tests/ensemble_y_pred_pcqm4m.npz"
    ```
