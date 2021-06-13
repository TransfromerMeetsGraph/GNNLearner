import os
import io
import argparse


god_num = 0.0027211385049999842


def main(args):
    assert os.path.exists(args.input)
    all_float_labels = io.open(args.input, "r", newline="\n", encoding="utf8").readlines()
    all_float_labels = [float(x.strip()) for x in all_float_labels]
    all_float_labels = [x / god_num for x in all_float_labels]
    all_int_labels = [int(x) for x in all_float_labels]
    if args.not_one_by_one:
        raise NotImplementedError()
    else:
        min_label = min(all_int_labels)
        max_label = max(all_int_labels)
        all_labels = list(range(min_label, max_label + 1))
        numel = len(all_labels)

    dirname = os.path.dirname(args.input)
    dictpath = os.path.join(dirname, "dict.{}.txt".format(numel))
    with io.open(dictpath, "w", encoding="utf8", newline="\n") as tgt:
        for label in all_labels:
            print("{} 100".format(label), file=tgt)

    for subset in ["train", "valid"]:
        label_fn = os.path.join(dirname, "{}.y".format(subset))
        label_new_fn = os.path.join(dirname, "{}.{}.y".format(subset, numel))
        raw = io.open(label_fn, "r", newline="\n", encoding="utf8").readlines()
        label_new = [int(float(x.strip()) / god_num) for x in raw]
        label_new = ["{}\n".format(x) for x in label_new]
        io.open(label_new_fn, "w", newline="\n", encoding="utf8").writelines(label_new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/blob/v-jinhzh/data/kddcup/raw/train.y")
    parser.add_argument("--not-one-by-one", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
