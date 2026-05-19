import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
import torch

try:
    from torch_linear_assignment import batch_linear_assignment
except Exception:
    batch_linear_assignment = None

_MATCH_BACKEND_LOGGED = False

def sinkhorn_norm(alpha: torch.Tensor, n_iter: int = 20) -> (torch.Tensor,):
    for _ in range(n_iter):
        alpha = alpha / alpha.sum(-1, keepdim=True)
        alpha = alpha / alpha.sum(-2, keepdim=True)
    return alpha

def log_sinkhorn_norm(log_alpha: torch.Tensor, n_iter: int =20) -> (torch.Tensor,):
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()

def gumbel_sinkhorn(log_alpha: torch.Tensor, tau: float = 1.0, n_iter: int = 20, noise: bool = True) -> (torch.Tensor,):
    if noise:
        uniform_noise = torch.rand_like(log_alpha)
        gumbel_noise = -torch.log(-torch.log(uniform_noise+1e-20)+1e-20)
        log_alpha = (log_alpha + gumbel_noise)/tau
    sampled_perm_mat = log_sinkhorn_norm(log_alpha, n_iter)
    return sampled_perm_mat

def gen_assignment(cost_matrix):
    row, col = linear_sum_assignment(cost_matrix)
    np_assignment_matrix = coo_matrix((np.ones_like(row), (row, col))).toarray()
    return np_assignment_matrix

def gumbel_matching(log_alpha : torch.Tensor, noise: bool = True) -> (torch.Tensor,):
    if noise:
        uniform_noise = torch.rand_like(log_alpha)
        gumbel_noise = -torch.log(-torch.log(uniform_noise+1e-20)+1e-20)
        log_alpha = (log_alpha + gumbel_noise)
    cost = -log_alpha
    global _MATCH_BACKEND_LOGGED
    use_cuda_backend = batch_linear_assignment is not None and cost.is_cuda
    assignment = None
    if use_cuda_backend:
        try:
            assignment = batch_linear_assignment(cost.contiguous())
        except RuntimeError:
            use_cuda_backend = False
            if not _MATCH_BACKEND_LOGGED:
                print(
                    "[gumbel_matching] backend=scipy_linear_sum_assignment device=cuda "
                    "reason=cuda_backend_error"
                )
                _MATCH_BACKEND_LOGGED = True
    if use_cuda_backend:
        if not _MATCH_BACKEND_LOGGED:
            print("[gumbel_matching] backend=torch_linear_assignment device=cuda")
            _MATCH_BACKEND_LOGGED = True
        B, n, m = cost.shape
        assignment_matrices = torch.zeros((B, n, m), device=cost.device, dtype=log_alpha.dtype)
        row_idx = torch.arange(n, device=cost.device).view(1, n).expand(B, n)
        batch_idx = torch.arange(B, device=cost.device).view(B, 1).expand(B, n)
        valid = assignment >= 0
        if valid.any():
            assignment_matrices[batch_idx[valid], row_idx[valid], assignment[valid]] = 1.0
        return assignment_matrices
    if not _MATCH_BACKEND_LOGGED:
        reasons = []
        if batch_linear_assignment is None:
            reasons.append("torch_linear_assignment_unavailable")
        if not cost.is_cuda:
            reasons.append("cpu_tensor")
        reason_str = ",".join(reasons) if reasons else "unknown"
        device_str = "cuda" if cost.is_cuda else "cpu"
        print(f"[gumbel_matching] backend=scipy_linear_sum_assignment device={device_str} reason={reason_str}")
        _MATCH_BACKEND_LOGGED = True
    np_log_alpha = log_alpha.detach().to("cpu").numpy()
    np_assignment_matrices = [gen_assignment(-x) for x in np_log_alpha]
    np_assignment_matrices = np.stack(np_assignment_matrices, 0)
    assignment_matrices = torch.from_numpy(np_assignment_matrices).float().to(log_alpha.device)
    return assignment_matrices

def inverse_permutation(X, permutation_matrix):
    return torch.einsum("bpq,bp->bq", (permutation_matrix, X))

def inverse_permutation_for_image(X, permutation_matrix):
    return torch.einsum("bpq,bpchw->bqchw", (permutation_matrix, X)).contiguous()
