import argparse
import io 
import os 
from sklearn.metrics import mean_absolute_error
import numpy as np
from molecule.inference import blob_path


def read_fn(fn):
    lines = io.open(fn, "r", newline="\n", encoding="utf8").readlines()
    lines = [float(x.strip()) for x in lines]
    return lines 

def main(args):
    fns = args.fns 
    assert len(fns) >= 1
    assert all(os.path.exists(fn) for fn in fns)
    preds = [np.array(read_fn(fn)).reshape(1, -1) for fn in fns]
    preds = np.concatenate(preds, axis=0)
    preds = np.mean(preds, axis=0)
    preds = preds.tolist()
    float_labels = "/blob/yinxia/wu2/data/molecule/ogb_data_processor/{}.y".format(args.subset)
    float_labels = blob_path(float_labels)
    float_labels = io.open(float_labels, "r", newline="\n", encoding="utf8").readlines()
    float_labels = [float(x.strip()) for x in float_labels]
    if args.subset in ["valid"]:
        mae = mean_absolute_error(float_labels, preds)
        print("mae: {}".format(mae))

    if args.output is not None:
        savedir = "/blob/v-jinhzh/model/pretrainmol/checkpoints/kddcup/pred"
        savefn = os.path.join(
            blob_path(savedir), "byol.ens1.pred.{}.{}".format(args.output, args.subset)
        )
        with io.open(savefn, "w", newline="\n", encoding="utf8",) as tgt:
            for p in preds:
                print(p, file=tgt)
        print("Saved to {}".format(savefn))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fns", type=str, nargs="+")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--subset", type=str, default="valid")
    args = parser.parse_args()
    main(args)