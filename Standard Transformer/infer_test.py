import os
import os.path as osp
from tqdm import tqdm

import click
import torch
import numpy as np

from fairseq.models.roberta import RobertaModel
from ext import KDDRobertaModel, KDDRobertaMBModel


@click.command()
@click.argument('data', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('folder', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('model_name', type=str)
@click.option('--bsz', type=int, default=32)
def process(data, folder, model_name, bsz):
    model = RobertaModel.from_pretrained(
        folder,
        checkpoint_file=model_name,
        data_name_or_path=data
    )

    model.cuda()
    model.eval()

    testset = model.task.load_dataset('test')
    testdata = list(testset)

    pred = []
    with torch.no_grad():
        n = len(testset)

        for i in tqdm(range(0, n, bsz)):
            batch = testset.collater(testdata[i:i+bsz])
            batch_pred = model.predict('sentence_classification_head', batch['net_input']['src_tokens'], True)
            pred.append(batch_pred)

    pred = torch.cat(pred, dim=0)
    
    pred = np.array(pred.squeeze(-1).cpu()).astype(np.float32)

    input_dict = {'y_pred': pred}
    save_test_submission(input_dict, model_name)


def save_test_submission(input_dict, model_name):
    '''
        save test submission file at dir_path
    '''
    assert('y_pred' in input_dict)
    y_pred = input_dict['y_pred']
 
    filename = "predictions/" + model_name
    assert(isinstance(filename, str))
    assert(isinstance(y_pred, np.ndarray) or isinstance(y_pred, torch.Tensor))
    assert(y_pred.shape == (377423,))

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    y_pred = y_pred.astype(np.float32)
    np.savez_compressed(filename, y_pred = y_pred)



if __name__ == '__main__':
    process()

