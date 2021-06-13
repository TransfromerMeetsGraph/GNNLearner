#!/usr/bin/env bash
set -x
set -e

cd /tmp/pretrainmol
MAX_SENTENCES=64
total=3045360
dev=false
epoch=50
UPDATEFREQ=1
LR=2e-4
datatype="tt"
dropout=0.1
relu=0
pooler=0
fp16=false

if [ -d /blob2 ]; then
    BLOB=/blob2
else
    BLOB=/blob
fi

cuda=0,1,2,3,4,5,6,7

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key=$1
    case $key in
    -b | --bsz)
        MAX_SENTENCES=$2
        shift 2
        ;;
    --epoch)
        epoch=$2
        shift 2
        ;;
    --uf)
        UPDATEFREQ=$2
        shift 2
        ;;
    --lr)
        LR=$2
        shift 2
        ;;
    --datatype)
        datatype=$2
        shift 2
        ;;
    -c | --cuda)
        cuda=$2
        shift 2
        ;;
    --dropout)
        dropout=$2
        shift 2
        ;;
    --relu)
        relu=$2
        shift 2
        ;;
    --pooler)
        pooler=$2
        shift 2
        ;;
    --fp16)
        fp16=true
        shift
        ;; 
    --dev)
        dev=true
        total=3426030
        shift
        ;;
    *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done

deno=$((MAX_SENTENCES * UPDATEFREQ * 8))
TOTAL_NUM_UPDATES=$(((total * epoch + deno - 1) / deno))
deno=$((deno * 100))
WARMUP_UPDATES=$(((total * epoch * 6 + deno - 1) / deno))

SAVE_DIR=${BLOB}/v-jinhzh/model/pretrainmol/checkpoints/kddcup/kcbyol_${datatype}_bs${MAX_SENTENCES}_epoch${epoch}_uf${UPDATEFREQ}_lr${LR}
SAVE_DIR=${SAVE_DIR}_d${dropout}_rd${relu}_pd${pooler}_fp16${fp16}_dev${dev}
SUFFIX=$(echo ${POSITIONAL[*]} | sed -r 's/-//g' | sed -r 's/\s+/_/g')
if [ -n "$SUFFIX" ]; then
    SAVE_DIR=${SAVE_DIR}_${SUFFIX}
fi
mkdir -p $SAVE_DIR
if [ "$dev" == "true" ]; then 
    DATADIR=${BLOB}/v-jinhzh/data/kddcup/bindatadev
else
    DATADIR=${BLOB}/v-jinhzh/data/kddcup/bindata
fi 

cudacap=$(python -c "import torch;print(torch.cuda.get_device_capability(0)[0] >= 7)")
if [ "$cudacap" == 'True' -a "$fp16" == "true" ]; then
    FP16="--fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128"
    # FP16=''
else
    FP16=''
fi

CUDA_VISIBLE_DEVICES=$cuda fairseq-train $DATADIR \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --task kddcup \
    --required-batch-size-multiple 1 \
    --arch doublemodel \
    --criterion graph_sp \
    --dropout $dropout --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    $FP16 --max-epoch $epoch \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --update-freq $UPDATEFREQ \
    --save-dir $SAVE_DIR \
    --pooler-dropout $pooler --relu-dropout $relu \
    --datatype $datatype --scaler-label ${POSITIONAL[@]} | tee -a $SAVE_DIR/training.log
