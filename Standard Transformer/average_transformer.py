import numpy as np

predictions = [
"l1_loss_reg_comb_seed4444.checkpoint119.pt.npz",
"l1_loss_reg_comb_seed88888888.checkpoint119.pt.npz",
"l1_loss_reg_comb_seed22.checkpoint119.pt.npz",
"l1_loss_reg_comb_seed666666.checkpoint119.pt.npz"
]

arr = 0

for p in predictions:
    X = np.load("predictions/" + p)
    arr += X['y_pred']

arr /= len(predictions)

np.savez_compressed("predictions/y_pred_pcqm4m_comb_test.npz", y_pred = arr.astype(np.float32))
