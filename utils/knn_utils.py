import torch


def square_distance(
    p1: torch.Tensor,
    p2: torch.Tensor
    ):
    """
    Calculate p1's pointwise square distance w.r.t p2
    """
    B, N, _ = p1.shape
    M = p2.shape[1]
    p1 = p1.unsqueeze(-2).repeat(1, 1, M, 1)
    p2 = p2.unsqueeze(1).repeat(1, N, 1, 1)
    square_dist = torch.sum((p1 - p2) ** 2, dim=-1)
    
    return square_dist


def knn(ref, query, k=1):
    """ 
    Compute k nearest neighbors for each query point.
    
    Params:
        ref: (B, 3, N)
        
        query: (B, 3, M)
    """
    device = ref.device
    ref = ref.float().transpose(2, 1).to(device)
    query = query.float().transpose(2, 1).to(device)
    dist = square_distance(query, ref)
    _, inds = torch.topk(dist, k, dim=-1, largest=False)
    inds = inds.transpose(2, 1)
    
    return inds