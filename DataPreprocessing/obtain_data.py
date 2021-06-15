import sys
import os

from ogb.lsc import PCQM4MDataset

if len(sys.argv) <= 1:
    print('Usage: python obtain_data.py path_to_smiles_datasets')
    exit(1)

output_path = sys.argv[1]
os.makedirs(output_path, exist_ok=True)

dataset = PCQM4MDataset(root="/tmp/ogb", only_smiles=True)

split_dict = dataset.get_idx_split()

train_dataset = [dataset[i] for i in split_dict['train']]
valid_dataset = [dataset[i] for i in split_dict['valid']]
test_dataset = [dataset[i] for i in split_dict['test']]

def writer(prefix, dump_list):
    fw_x = open(output_path + '/' + prefix + ".x", "w", encoding="utf8")
    fw_y = open(output_path + '/' + prefix + ".y", "w", encoding="utf8")
    
    for (x,y) in dump_list:
        print(x.strip(), file=fw_x)
        print(y, file=fw_y)

    fw_x.close()
    fw_y.close()

writer("train", train_dataset)
writer("valid", valid_dataset)
writer("test", test_dataset)
