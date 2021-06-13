import io
import os
import argparse

god_num = 0.0027211385049999842


def main(input_fn, output, dictionary):
    all_int_labels = io.open(dictionary, "r", newline="\n", encoding="utf8").readlines()
    all_int_labels = [int(x.strip().split()[0]) for x in all_int_labels]
    min_label = all_int_labels[0]
    max_label = all_int_labels[-1]

    new_float = io.open(input_fn, "r", newline="\n", encoding="utf8").readlines()
    new_float = [float(x.strip()) for x in new_float]
    new_int = [int((f + god_num / 2) / god_num) for f in new_float]
    new_int = [x if x > min_label else min_label for x in new_int]
    new_int = [x if x < max_label else max_label for x in new_int]
    new_float = ["{}\n".format(x) for x in new_float]
    new_int = ["{}\n".format(x) for x in new_int]

    output_cls = "{}.cls".format(output)
    # output_cls = os.path.join(output, "train.cls.label")
    io.open(output_cls, "w", newline="\n", encoding="utf8").writelines(new_int)
    output_reg = "{}.reg".format(output)
    # output_reg = os.path.join(output, "train.reg.label")
    io.open(output_reg, "w", newline="\n", encoding="utf8").writelines(new_float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/tmp/valid.y")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dict", type=str, default="/blob/v-jinhzh/data/kddcup/raw/dict.17144.txt")
    args = parser.parse_args()
    input_fn = args.input
    assert os.path.exists(input_fn)
    output = args.output
    if output is None:
        output = input_fn
    dictionary = args.dict
    assert os.path.exists(dictionary)
    main(input_fn, output, dictionary)