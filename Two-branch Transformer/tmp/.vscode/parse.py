import re

args = r"data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args  \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric"

patter = re.compile('(?:[\'\"])(.+)(?:[\'\"])|(\S+)')
args = re.sub(r'\\', "", args)
tokens = patter.findall(args)
tokens = [t[0] if t[0] else t[1] for t in tokens]
tokens = ['"{}"'.format(t) for t in tokens]
args = ','.join(tokens)
print(args)