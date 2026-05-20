
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def earth_mover_distance(pc1, pc2, transpose=False):
    if transpose:
        pc1 = pc1.transpose(-1, -2)
        pc2 = pc2.transpose(-1, -2)
    
    if pc1.dim() == 2:
        pc1 = pc1.unsqueeze(0)
    if pc2.dim() == 2:
        pc2 = pc2.unsqueeze(0)
    
    batch_size = pc1.size(0)
    device = pc1.device
    
    pc1_np = pc1.detach().cpu().numpy()
    pc2_np = pc2.detach().cpu().numpy()
    
    emd_values = []
    
    for b in range(batch_size):
        emd = _compute_emd_single(pc1_np[b], pc2_np[b])
        emd_values.append(emd)
    
    return torch.tensor(emd_values, device=device, dtype=pc1.dtype)


def _compute_emd_single(pc1, pc2):
    n1, n2 = len(pc1), len(pc2)
    
    if n1 == 0 or n2 == 0:
        return 0.0
    
    if n1 != n2:
        if n1 > n2:
            indices = np.random.choice(n2, n1, replace=True)
            pc2 = pc2[indices]
        else:
            indices = np.random.choice(n1, n2, replace=True)
            pc1 = pc1[indices]
    
    cost_matrix = cdist(pc1, pc2, metric='euclidean')
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    emd = cost_matrix[row_ind, col_ind].mean()
    
    return emd
