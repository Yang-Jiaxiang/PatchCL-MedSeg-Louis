import torch 
import numpy as np
from utils.hard_neg_mining import hard_neg, simple_hard_neg

def simple_PCGJCL(patch_list, embd_queues, emb_dim, tau, lamb, psi=4096):
    """
    INPUT:
    patch_list = list of length num_class_in_patches containing tensors t_i
    t_i = tensor of shape num_samples_class_i * dim
    tau = temperature parameter
    lamb = lambda : scaling parameter
    alpha = negative weighting term 

    OUTPUT:
    loss = tensor
    """
    N = len(patch_list)
    total_samples = 0
    mean_list = []
    cov_list = []
    loss = torch.tensor([0.0], dtype=torch.float32)
    num_classes_in_batch = 0

    # 計算每個類別的均值和協方差矩陣
    for i in range(N):
        if patch_list[i] is not None:
            t_i = torch.stack(embd_queues[i], dim=0)
            total_samples += patch_list[i].size(0)
            num_classes_in_batch += 1
            mu = torch.mean(t_i, dim=0)
            mean_list.append(mu)
            sig = torch.mm((t_i - mu).t(), (t_i - mu)) / t_i.size(0)
            cov_list.append(sig)
        else:
            mean_list.append(None)
            cov_list.append(None)

    g_count = 0
    den_lists = [[] for _ in range(N)]
    pos_lists = [None for _ in range(N)]
    
    for i in range(N):
        if patch_list[i] is not None:
            t_i, mu_i, sig_i = patch_list[i], mean_list[i], cov_list[i]
            num = (-torch.sum(torch.mm(t_i, mu_i.view(emb_dim, 1)), dim=0)) / tau
            l_count = 0
            # 遍歷所有其他類別，計算負樣本對比損失
            for j in range(N):
                if patch_list[j] is not None and i != j:
                    t_j, mu_j, sig_j = torch.stack(embd_queues[j], dim=0), mean_list[j], cov_list[j]
                    # 使用簡化的 hard neg 函數
                    hn_mu_j, hn_sig_j = simple_hard_neg(t_i, mu_i, t_j, mu_j, 0.5, sample_size=500)
                    
                    if hn_mu_j is None or hn_sig_j is None:
                        print(f"Skipping class pair ({i}, {j}) due to lack of negative samples.")
                        continue

                    den_neg = ((torch.mm(t_i, hn_mu_j.view(emb_dim, 1))) / tau) + (0.5 * lamb / (tau ** 2)) * (torch.diag(torch.mm(t_i, torch.mm(hn_sig_j, t_i.t()))).view(-1, 1))
                    den_lists[i].append(den_neg)
                elif patch_list[j] is not None:
                    den_pos = torch.mm(t_i, mu_i.view(emb_dim, 1)) / tau + (0.5 * lamb / (tau ** 2)) * (torch.diag(torch.mm(t_i, torch.mm(sig_i, t_i.t()))).view(-1, 1))
                    pos_lists[i] = den_pos
        
            a = pos_lists[i]
            res = torch.zeros(a.size())
            res += torch.exp(a)
            for d in den_lists[i]:
                res += (psi / num_classes_in_batch) * torch.exp(d)
            den = torch.sum(torch.log(res), dim=0)
            if g_count == 0:
                loss = num + den
                g_count += 1
            else:
                loss += num + den

    loss = loss / total_samples
    return loss


def PCGJCL(patch_list, embd_queues, emb_dim, tau, lamb, psi=4096):
  
    """
    INPUT :
    patch_list = list of length num_class_in_patches containing tensors t_i
    t_i = tensor of shape num_samples_class_i*dim
    tau = temperature parameter
    lamb = lambda : scaling parameter
    alpha = negative weighting term 

    OUTPUT :
    loss = tensor
    """
    N= len(patch_list)
    total_samples=0
    mean_list=[]
    cov_list=[]
    loss = torch.tensor([0])
    num_classes_in_batch=0
    #calculate mean and cov matrices for each class 
    for i in range(N):
        if patch_list[i] is not None:
            t_i = torch.stack(embd_queues[i], dim=0)
            total_samples += patch_list[i].size()[0]
            num_classes_in_batch+=1
            mu=torch.mean(t_i, dim=0)
            mean_list.append(mu)
            sig =torch.mm((t_i-mu).t(), (t_i-mu))/(t_i.size()[0])
            cov_list.append(sig)
        else:
            mean_list.append(None)
            cov_list.append(None)

    g_count=0
    den_lists=[]
    pos_lists=[]
    for i in range(N):
        if patch_list[i] is not None:
            t_i, mu_i, sig_i=patch_list[i], mean_list[i], cov_list[i]
            num = (-torch.sum(torch.mm(t_i, mu_i.view(emb_dim, 1)), dim=0))/tau
            #den_neg, den_pos = torch.tensor([0]), torch.tensor([0])
            l_count=0
            #Iterate over neg classes for a particluar class
            for j in range(N):
                if patch_list[j] is not None:
                    if i!=j:
                        t_j, mu_j, sig_j= torch.stack(embd_queues[j], dim=0), mean_list[j], cov_list[j]
                        #get hard neg mu_j,sig_j
                        hn_mu_j, hn_sig_j = hard_neg(t_i, mu_i, sig_i, t_j, mu_j, sig_j, 0.5)# GET HARD NEGS HERE
                        #hn_mu_j, hn_sig_j = mu_j, sig_j # USE THIS FOR NO HARD NEGS
                        if l_count==0:
                            den_neg = ((torch.mm(t_i, hn_mu_j.view(emb_dim, 1)))/tau) + (0.5*lamb/(tau**2))*(torch.diag((torch.mm(t_i, torch.mm(hn_sig_j, t_i.t())))).view(-1,1))
                            l_count+=1
                            den_lists.append([den_neg])
                        else:
                            den_neg = ((torch.mm(t_i, hn_mu_j.view(emb_dim, 1)))/tau) + (0.5*lamb/(tau**2))*(torch.diag((torch.mm(t_i, torch.mm(hn_sig_j, t_i.t())))).view(-1,1))
                            den_lists[i].append(den_neg)
                
                    else:
                        den_pos = torch.mm(t_i, mu_i.view(emb_dim, 1))/tau + (0.5*lamb/(tau**2))*(torch.diag((torch.mm(t_i, torch.mm(sig_i, t_i.t())))).view(-1,1))
                        pos_lists.append(den_pos)
                # else:
                #     den_lists[i].append(None)
        
            a=pos_lists[i]
            res=torch.zeros(a.size())
            res+=torch.exp(a)       
            for d in den_lists[i]:
              res+=(psi/num_classes_in_batch)*torch.exp(d)# ADDED WEIGHT FOR NEG TERMS HERE ....psi*den_neg+den_pos
            den = torch.sum(torch.log(res), dim=0)
            if g_count==0:
                loss= num+den
                g_count+=1
            else:
                loss+=num+den
        
        else:
          pos_lists.append(None)
          den_lists.append(None)

    loss = loss/total_samples
    
    return loss


# +
import torch

def PCGJCL_GPU(patch_list, embd_queues, emb_dim, tau, lamb, psi=4096):
    """
    INPUT :
    patch_list = list of length num_class_in_patches containing tensors t_i
    t_i = tensor of shape num_samples_class_i*dim
    tau = temperature parameter
    lamb = lambda : scaling parameter
    alpha = negative weighting term 

    OUTPUT :
    loss = tensor
    """
    N = len(patch_list)
    total_samples = 0
    mean_list = []
    cov_list = []
    device = patch_list[0].device if patch_list[0] is not None else 'cpu'
    loss = torch.tensor(0.0, device=device)
    num_classes_in_batch = 0

    # Calculate mean and covariance matrices for each class 
    for i in range(N):
        if patch_list[i] is not None:
            t_i = torch.stack(embd_queues[i], dim=0)
            total_samples += patch_list[i].size(0)
            num_classes_in_batch += 1
            mu = torch.mean(t_i, dim=0)
            mean_list.append(mu)
            sig = torch.mm((t_i - mu).t(), (t_i - mu)) / t_i.size(0)
            cov_list.append(sig)
        else:
            mean_list.append(None)
            cov_list.append(None)

    pos_lists = []
    den_lists = [[] for _ in range(N)]

    for i in range(N):
        if patch_list[i] is not None:
            t_i, mu_i, sig_i = patch_list[i], mean_list[i], cov_list[i]
            num = -torch.sum(torch.mm(t_i, mu_i.view(emb_dim, 1)), dim=0) / tau
            den_pos = (torch.mm(t_i, mu_i.view(emb_dim, 1)) / tau + 
                       (0.5 * lamb / (tau ** 2)) * torch.diag(torch.mm(t_i, torch.mm(sig_i, t_i.t()))).view(-1, 1))
            pos_lists.append(den_pos)

            for j in range(N):
                if patch_list[j] is not None and i != j:
                    t_j, mu_j, sig_j = torch.stack(embd_queues[j], dim=0), mean_list[j], cov_list[j]
                    hn_mu_j, hn_sig_j = hard_neg(t_i, mu_i, sig_i, t_j, mu_j, sig_j, 0.5)
                    den_neg = (torch.mm(t_i, hn_mu_j.view(emb_dim, 1)) / tau + 
                               (0.5 * lamb / (tau ** 2)) * torch.diag(torch.mm(t_i, torch.mm(hn_sig_j, t_i.t()))).view(-1, 1))
                    den_lists[i].append(den_neg)

    for i in range(len(pos_lists)):
        a = pos_lists[i]
        res = torch.exp(a)
        for d in den_lists[i]:
            res = res + (psi / num_classes_in_batch) * torch.exp(d)
        den = torch.sum(torch.log(res), dim=0)
        loss = loss + num + den

    loss = loss / total_samples
    
    return loss
