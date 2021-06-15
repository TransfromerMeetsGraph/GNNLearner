import numpy as np
import sys
import os.path as osp
import os

def save_test_submission(input_dict, dir_path):
    '''
        save test submission file at dir_path
    '''
    assert('y_pred' in input_dict)
    y_pred = input_dict['y_pred']

    if not osp.exists(dir_path):
        os.makedirs(dir_path)
        
    filename = osp.join(dir_path, 'y_pred_pcqm4m')
    assert(isinstance(filename, str))
    assert(isinstance(y_pred, np.ndarray))
    assert(y_pred.shape == (377423,))

   
    y_pred = y_pred.astype(np.float32)
    np.savez_compressed(filename, y_pred = y_pred)


def read_numpy(fname):
    return np.load(fname)["y_pred"]
    
def read_txt(fname):
    with open(fname, "r", encoding="utf8") as fr:
        preds = [float(e.strip()) for e in fr]
    return np.array(preds).astype(np.float32)
    
def read(fname):
    if fname.endswith(".npz") or fname.endswith(".npy"):
        return read_numpy(fname)
    return read_txt(fname)

input_files = [
"byol.reg.pred.kcbyol_tt_bs64_epoch50_uf1_lr2e-4_d0.1_rd0_pd0_fp16false_deva_usebyol.test",
"byol.reg.pred.kcbyol_tt_bs64_epoch50_uf1_lr2e-4_d0.1_rd0_pd0_fp16false_devb_usebyol.test",
"byol.reg.pred.kcbyol_tt_bs64_epoch50_uf1_lr2e-4_d0.1_rd0_pd0_fp16false_devc_usebyol.test",
"byol.reg.pred.kcbyol_tt_bs64_epoch50_uf1_lr2e-4_d0.1_rd0_pd0_fp16false_devd_usebyol.test",
"byol.reg.pred.kcbyol_tt_bs64_epoch50_uf1_lr2e-4_d0.1_rd0_pd0_fp16false_devtrue_usebyol_seed_0.test",
"byol.reg.pred.kcbyol_tt_bs64_epoch50_uf1_lr2e-4_d0.1_rd0_pd0_fp16false_devtrue_usebyol_seed_1.test",
"byol.reg.pred.kcbyol_tt_bs64_epoch50_uf1_lr2e-4_d0.1_rd0_pd0_fp16false_devtrue_usebyol_seed_2.test",
"byol.reg.pred.kcbyol_tt_bs64_epoch50_uf1_lr2e-4_d0.1_rd0_pd0_fp16false_devtrue_usebyol_seed_3.test",
"test_t-yafan_avg-gnn-top5_nan.npz",
"y_pred_pcqm4m_comb_test.npz"
]

input_weights = [
1,1,1,1,
1.25,1.25,1.25,1.25,
2,1
]

assert len(input_files) == len(input_weights)


ret = 0
for (w, ff) in zip(input_weights, input_files):
    print(f"Dealing with {ff}\t{float(w) / sum(input_weights)}")
    ret += float(w) / sum(input_weights) * read(ff)
    

save_test_submission({'y_pred': ret}, "winner")





