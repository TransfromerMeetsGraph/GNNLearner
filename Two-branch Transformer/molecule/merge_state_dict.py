import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--nmt",
    type=str,
    default="/blob/v-jinhzh/model/pretrainmol/checkpoints/retrosys/uspto50k/checkpoint100.pt",
)
parser.add_argument(
    "--plm",
    type=str,
    default="/blob/v-jinhzh/model/pretrainmol/checkpoints/bntg-pubchem-10m-doublemodel-tu-125000-wu-10000-lr-0.0005-uf-16-mt-12288-usemlm-usecontrastive-usebottleneckhead-bottleneckratio-4/checkpoint41.pt",
)
args = parser.parse_args()


def load_state_from_ckt(fn):
    assert os.path.exists(fn), "{} does not exist!".format(fn)
    return torch.load(fn, map_location=torch.device("cpu"))


nmt_state = load_state_from_ckt(args.nmt)
plm_state = load_state_from_ckt(args.plm)

for k, v in plm_state['model'].items():
    nmt_state['model']["encoder.plm_encoder.{}".format(k)] = v

save_fn = "{}.plm".format(args.nmt)
torch.save(nmt_state, save_fn)
print("save to {}".format(save_fn))