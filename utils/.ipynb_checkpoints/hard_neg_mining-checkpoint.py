import random
import numpy as np
import torch
import torch.nn.functional as F

def hard_neg(samples_p, mean_p, cov_p, samples_n, mean_n, cov_n, beta=0.5):
    """
    INPUT:
    t_p = tensor of shape num_examples_of_positive_class*embedding_dimension
    mu_p = tensor denoting mean of positive class
    sig_p = tensor denoting cov matrix of positive class
    t_n = tensor of shape num_examples_of_negative_class*embedding_dimension 
    mu_n = tensor denoting mean of negative class
    sig_n = tensor denoting cov matrix of negative class
    beta = int random weight for convex combination sampled from Uniform (0, beta)

    OUTPUT:
    (hn_mu_n, hn_sig_n)=(mean, cov)[tensors] of distributiion incoporating hard negative mining
    """
    num_pos, num_neg = samples_p.size()[0], samples_n.size()[0]
    with torch.no_grad():
        t_p, mu_p, sig_p, t_n, mu_n, sig_n = samples_p.numpy(), mean_p.numpy(), cov_p.numpy(), samples_n.numpy(), mean_n.numpy(), cov_n.numpy()
        sig_p_inv, sig_n_inv = np.linalg.pinv(sig_p), np.linalg.pinv(sig_n)
        sig_p_inv, sig_n_inv = sig_p_inv/1e10, sig_n_inv/1e10
        # calculate mahalanobis dist and find anchors
        dists_from_p = np.diag(np.matmul((t_n-mu_p),np.matmul(sig_p_inv,(t_n-mu_p).T)))
        anchor_n = t_n[np.argmin(dists_from_p),:]
        dists_from_n = np.diag(np.matmul((t_p-mu_n),np.matmul(sig_n_inv,(t_p-mu_n).T)))
        anchor_p = t_p[np.argmin(dists_from_n),:]
        # define normal vector
        normal_vec=anchor_n-anchor_p
        # define constraints
        l, u = np.dot(normal_vec, anchor_p), np.dot(normal_vec, anchor_n)

        # sample till you get 3*num_neg negatives satisfying constraints
        negs=np.ndarray(0)
        while (True) :
            sample = np.random.multivariate_normal(mu_n, sig_n, 1000)
            checker = np.dot(sample, normal_vec.T)
            sample = sample[np.logical_and(checker<=u,checker>=l)]
            if negs.shape[0]==0:
                negs = sample
            else :
                negs = np.vstack((negs, sample))
            # break out of loop when you get enough samples
            if negs.shape[0]>=3*num_neg:
                break
        
        # sort samples acc to mahalanobis dist to neg distribution
        dists = np.diag(np.matmul((negs-mu_n),np.matmul(sig_n_inv,(negs-mu_n).T)))
        negs = negs[np.argsort(dists)] #sorts sampled points in ascedning order of mahalanobis dist from negative distribution

        # extract top num_neg samples for checking distance constraint to obtain qualified negs
        sampled_negs = negs[0:num_neg]
        dist_n_achor = np.matmul((anchor_n-mu_p),np.matmul(sig_p_inv,(anchor_n-mu_p).T))
        dist_of_sampled_from_p = np.diag(np.matmul((sampled_negs-mu_p),np.matmul(sig_p_inv,(sampled_negs-mu_p).T)))
        disqualified_indices=dist_of_sampled_from_p>dist_n_achor
        disqualified_negs = sampled_negs[disqualified_indices]
        inter_negs=sampled_negs#intialising
        for i in range(sum(disqualified_indices)):
            w= np.random.uniform(beta-0.2,beta+0.2)
            temp=random.choices(inter_negs, k=2)
            new_neg = w*temp[0]+(1-w)*temp[1]
            inter_negs=np.vstack((new_neg, inter_negs))
    all_negs=torch.from_numpy(inter_negs)
    # NORMALIZE EMBEDDINGS WITH L2 NORM=1
    all_negs = F.normalize(all_negs, p=2, dim=1)
    hn_mu_n = torch.mean(all_negs, dim=0)
    hn_sig_n = torch.mm((all_negs-hn_mu_n).t(), ((all_negs-hn_mu_n)))/(all_negs.size()[0])

    return hn_mu_n.float(), hn_sig_n.float()


# +
import torch
import torch.nn.functional as F
import numpy as np

def hard_neg_GPU(samples_p, mean_p, cov_p, samples_n, mean_n, cov_n, beta=0.5):
    """
    INPUT:
    samples_p = tensor of shape num_examples_of_positive_class * embedding_dimension
    mean_p = tensor denoting mean of positive class
    cov_p = tensor denoting cov matrix of positive class
    samples_n = tensor of shape num_examples_of_negative_class * embedding_dimension 
    mean_n = tensor denoting mean of negative class
    cov_n = tensor denoting cov matrix of negative class
    beta = int random weight for convex combination sampled from Uniform (0, beta)

    OUTPUT:
    (hn_mu_n, hn_sig_n) = (mean, cov) [tensors] of distribution incorporating hard negative mining
    """
    device = samples_p.device

    # Calculate inverse covariance matrices using torch
    sig_p_inv = torch.linalg.pinv(cov_p) / 1e10
    sig_n_inv = torch.linalg.pinv(cov_n) / 1e10

    # Calculate Mahalanobis distance and find anchors
    dists_from_p = torch.einsum('ij,jk,ik->i', samples_n - mean_p, sig_p_inv, samples_n - mean_p)
    anchor_n = samples_n[torch.argmin(dists_from_p)]
    dists_from_n = torch.einsum('ij,jk,ik->i', samples_p - mean_n, sig_n_inv, samples_p - mean_n)
    anchor_p = samples_p[torch.argmin(dists_from_n)]

    # Define normal vector
    normal_vec = anchor_n - anchor_p
    l, u = torch.dot(normal_vec, anchor_p), torch.dot(normal_vec, anchor_n)

    # Sample negatives satisfying constraints
    negs = []
    while len(negs) < 3 * samples_n.size(0):
        samples = torch.distributions.multivariate_normal.MultivariateNormal(mean_n, cov_n).sample((1000,))
        checker = torch.dot(samples, normal_vec)
        valid_samples = samples[(checker <= u) & (checker >= l)]
        negs.extend(valid_samples)

    negs = torch.stack(negs[:3 * samples_n.size(0)], dim=0)
    dists = torch.einsum('ij,jk,ik->i', negs - mean_n, sig_n_inv, negs - mean_n)
    negs = negs[torch.argsort(dists)][:samples_n.size(0)]

    # Distance constraint
    dist_n_anchor = torch.dot(anchor_n - mean_p, torch.mv(sig_p_inv, anchor_n - mean_p))
    dist_of_sampled_from_p = torch.einsum('ij,jk,ik->i', negs - mean_p, sig_p_inv, negs - mean_p)
    disqualified_indices = dist_of_sampled_from_p > dist_n_anchor
    disqualified_negs = negs[disqualified_indices]
    inter_negs = negs

    for i in range(disqualified_negs.size(0)):
        w = torch.empty(1).uniform_(beta - 0.2, beta + 0.2).item()
        temp = torch.randint(0, inter_negs.size(0), (2,))
        new_neg = w * inter_negs[temp[0]] + (1 - w) * inter_negs[temp[1]]
        inter_negs = torch.vstack((new_neg, inter_negs))

    all_negs = inter_negs.to(device)
    # Normalize embeddings with L2 norm = 1
    all_negs = F.normalize(all_negs, p=2, dim=1)
    hn_mu_n = torch.mean(all_negs, dim=0)
    hn_sig_n = torch.mm((all_negs - hn_mu_n).t(), (all_negs - hn_mu_n)) / all_negs.size(0)

    return hn_mu_n.float(), hn_sig_n.float()
