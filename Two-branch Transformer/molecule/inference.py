import os
import argparse
import io
from fairseq.models.onemodel import OneModel
from sklearn.metrics import mean_absolute_error
import re

model_name = {0: "cls", 1: "reg", 2: "ens"}


def main(dataset, cktpath, subset, args):
    roberta = OneModel.from_pretrained(
        os.path.dirname(cktpath),
        checkpoint_file=os.path.basename(cktpath),
        data_name_or_path=dataset,
    )
    roberta.cuda()
    roberta.eval()

    assert subset in ["train", "valid", "test"]
    roberta.load_data(subset)

    if args.suffix is None:
        args.suffix = os.path.split(os.path.dirname(cktpath))[1]

    float_labels = args.label_fn
    if float_labels is not None:
        float_labels = blob_path(float_labels)
        float_labels = io.open(float_labels, "r", newline="\n", encoding="utf8").readlines()
        float_labels = [float(x.strip()) for x in float_labels]

    results = roberta.inference(split=subset)
    for i, output in enumerate(results):
        output = output.tolist()
        savedir = "/tmp/kddcup/pred"
        os.makedirs(savedir, exist_ok=True)
        savefn = os.path.join(
            blob_path(savedir), "byol.{}.pred.{}.{}".format(model_name[i], args.suffix, subset)
        )
        with io.open(savefn, "w", newline="\n", encoding="utf8",) as tgt:
            for p in output:
                print(p, file=tgt)
        print("Saved to {}".format(savefn))
        if float_labels is None:
            continue
        mae = mean_absolute_error(float_labels, output)
        print("MAE: {}".format(mae))


def blob_path(path):
    if os.path.isdir("/blob2"):
        return re.sub("/blob/", "/blob2/", path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/blob/v-jinhzh/data/kddcup/bindata")
    parser.add_argument("cktpath", type=str)
    parser.add_argument("--subset", type=str, default="valid")
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--label-fn", type=str, default=None)
    args = parser.parse_args()
    dataset = args.dataset
    dataset = blob_path(dataset)
    cktpath = args.cktpath
    subset = args.subset
    assert os.path.exists(cktpath)
    main(dataset, cktpath, subset, args)
