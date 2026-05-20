
import torch
import torch.nn as nn


class chamfer_3DDist(nn.Module):
    
    def __init__(self):
        super(chamfer_3DDist, self).__init__()
    
    def forward(self, pc1, pc2):
        return chamfer_distance(pc1, pc2)


def chamfer_distance(pc1, pc2):
    if pc1.dim() == 2:
        pc1 = pc1.unsqueeze(0)
    if pc2.dim() == 2:
        pc2 = pc2.unsqueeze(0)
    
    batch_size, n_points, dim = pc1.size()
    _, m_points, _ = pc2.size()
    
    
    pc1_sq = torch.sum(pc1 ** 2, dim=2, keepdim=True)
    
    pc2_sq = torch.sum(pc2 ** 2, dim=2, keepdim=True).transpose(1, 2)
    
    cross = torch.bmm(pc1, pc2.transpose(1, 2))
    
    dist_matrix = pc1_sq + pc2_sq - 2 * cross
    
    dist_matrix = torch.clamp(dist_matrix, min=0.0)
    
    dist1, idx1 = torch.min(dist_matrix, dim=2)
    
    dist2, idx2 = torch.min(dist_matrix, dim=1)
    
    return dist1, dist2, idx1, idx2
