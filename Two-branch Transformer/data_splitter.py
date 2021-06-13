import os
from string import ascii_lowercase

def worker(idx):
    X = "train.x.bpe"
    Y = "train.y"
    Z = "train.17144.y"
    for ch in ascii_lowercase[:4]:
        if ch == idx:
            continue
        X += f" valid.x.bpea{ch}"
        Y += f" valid.ya{ch}"
        Z += f" valid.17144.ya{ch}"

    os.system(f"cat {X} > train.aug.{idx}.x")
    os.system(f"cat {Y} > train.aug.{idx}.yr")
    os.system(f"cat {Z} > train.aug.{idx}.yc")
    os.system(f"cat valid.x.bpea{idx} > valid.aug.{idx}.x")
    os.system(f"cat valid.ya{idx} > valid.aug.{idx}.yr")
    os.system(f"cat valid.17144.ya{idx} > valid.aug.{idx}.yc")


os.system("split -l 95168 valid.y valid.y")
os.system("split -l 95168 valid.x.bpe valid.x.bpe")
os.system("split -l 95168 valid.17144.y valid.17144.y")

worker("a")
worker("b")
worker("c")
worker("d")