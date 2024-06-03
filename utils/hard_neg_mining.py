import random
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

def simple_hard_neg(samples_p, mean_p, samples_n, mean_n, beta=0.5, sample_size=500):
    """
    簡化版 hard_neg，使用歐氏距離計算困難負樣本
    INPUT:
    samples_p = 正樣本的張量，形狀為 (num_samples_p, embedding_dimension)
    mean_p = 正樣本的均值向量
    samples_n = 負樣本的張量，形狀為 (num_samples_n, embedding_dimension)
    mean_n = 負樣本的均值向量
    beta = 用於負樣本的凸組合權重
    sample_size = 每次生成的樣本數量

    OUTPUT:
    (hn_mu_n, hn_sig_n) = (均值, 協方差矩陣) 的張量
    """
    num_pos, num_neg = samples_p.size(0), samples_n.size(0)
    with torch.no_grad():
        # 將張量轉換為 NumPy 陣列
        t_p, mu_p, t_n, mu_n = samples_p.numpy(), mean_p.numpy(), samples_n.numpy(), mean_n.numpy()

        # 計算負樣本到正樣本均值的歐氏距離，並選擇距離最遠的樣本
        dists_from_p = np.linalg.norm(t_n - mu_p, axis=1)
        farthest_neg_idx = np.argmax(dists_from_p)
        farthest_neg = t_n[farthest_neg_idx]

        # 生成新的負樣本
        new_negs = np.random.uniform(mu_n, farthest_neg, size=(3 * num_neg, mu_n.shape[0]))
        
        if new_negs.size == 0:
#             print("No negative samples generated. Please check the input parameters and conditions.")
            return None, None

        # 轉換回 PyTorch 張量
        all_negs = torch.from_numpy(new_negs).float()
        all_negs = F.normalize(all_negs, p=2, dim=1)
        hn_mu_n = torch.mean(all_negs, dim=0)
        hn_sig_n = torch.mm((all_negs - hn_mu_n).t(), (all_negs - hn_mu_n)) / all_negs.size(0)

    return hn_mu_n, hn_sig_n