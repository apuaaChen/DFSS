"""
Entry point of SDDMM kernel
"""
import torch
from dfss.sddmm import sddmm_bf16_ntn
from typing import Tuple
import torch.nn.functional as F


def sddmm(lhs_matrix: torch.Tensor, rhs_matrix: torch.Tensor, mask = None, training=False) -> Tuple[torch.Tensor, torch.Tensor]:
    r""" A customized SDDMM kernel. It generates the nonzeros and metadata that matches the input
    format of cutlass SpMM with sparse tensor core. 
    With ``float32``  tensors, it will select the larger one from each 1x2 vector in the output.
    With ``bfloat16`` tensors, it will setect the larger two from each 1x4 vector in the output.

    It performs :math:`A x B^T \cdot C = D`, and the generates C & D

    Args: 
        `B`: batch size, (`M`, `K`, `N`) defines the GEMM size
        lhs_matrix: :math:`A` from an expected input of size `(B, M, K)` or `(M, K)`
        rhs_matrix: :math:`B` from an expected input of size `(B, N, K)` or `(N, K)`
        mask (Optional): An optional mask to be added to the GEMM output.
        training: Default: false, whether it is under the training mode
    
    Returns:
        nonzeros  : :math:`D` of size `(B, M, N/2)` or `(M, N/2)` 
        metadata  : :math:`C` of size `(B, M, N/Q)` or `(M, N/Q)`, `Q`=8 for `float32` and `Q`=16 for `bfloat16`
    
    Example:
        >>> import torch
        >>> import dspattn
        >>> lhs_matrix = torch.randn(size=(8, 4096, 64), dtype=torch.bfloat16, device='cuda')
        >>> rhs_matrix = torch.randn(size=(8, 4096, 64), dtype=torch.bfloat16, device='cuda')
        >>> nonzeros, metadata = dspattn.sddmm(lhs_matrix, rhs_matrix)
    """

    #########################
    # Check input dimension #
    #########################

    if lhs_matrix.dim() != 2 and lhs_matrix.dim() != 3:
        raise ValueError("expected 2D or 3D lhs matrix (got {}D input)".format(lhs_matrix.dim()))
    
    if rhs_matrix.dim() != 2 and rhs_matrix.dim() != 3:
        raise ValueError("expected 2D or 3D rhs matrix (got {}D input)".format(rhs_matrix.dim()))

    if lhs_matrix.dim() != rhs_matrix.dim():
        raise ValueError("the two input matrices must have the "
                         "same dimension (got {}D lhs_matrix and {}D rhs_matrix)".format(lhs_matrix.dim(), rhs_matrix.dim()))
    
    if lhs_matrix.dim() == 3 and lhs_matrix.size(0) != rhs_matrix.size(0):
        raise ValueError("the batch size should be the same (got B={} in lhs_matrix and B={} in rhs_matrix)".format(lhs_matrix.size(0), rhs_matrix.size(0)))

    if lhs_matrix.size(-1) != rhs_matrix.size(-1):
        raise ValueError("the reduction dim K should be the same (got K={} in lhs_matrix and K={} in rhs_matrix".format(lhs_matrix.size(-1), rhs_matrix.size(-1)))

    #########################
    # Check input data type #
    #########################

    if lhs_matrix.dtype != torch.float32 and lhs_matrix.dtype != torch.bfloat16:
        raise ValueError("the lhs_matrix should be in torch.float32 or torch.bfloat16 (got {})".format(lhs_matrix.dtype))
    
    if rhs_matrix.dtype != torch.float32 and rhs_matrix.dtype != torch.bfloat16:
        raise ValueError("the rhs_matrix should be in torch.float32 or torch.bfloat16 (got {})".format(rhs_matrix.dtype))

    if not lhs_matrix.is_cuda:
        raise ValueError("the lhs_matrix should be on GPU (got CPU)")
    
    if not rhs_matrix.is_cuda:
        raise ValueError("the rhs_matrix should be on GPU (got CPU)")

    ################################
    # launch the extended function #
    ################################

    # for float32 tensors
    if lhs_matrix.dtype == torch.float32:
        if training:
            # Emulate the SDDMM kernel with naive pytorch API
            if lhs_matrix.dim() == 2:
                dense_matrix = torch.matmul(lhs_matrix, torch.transpose(rhs_matrix, 0, 1)).unsqueeze(0)
            else:
                dense_matrix = torch.bmm(lhs_matrix, torch.transpose(rhs_matrix, 1, 2))
            
            if mask is not None:
                dense_matrix = dense_matrix.view((mask.size(0), -1, dense_matrix.size(1), dense_matrix.size(2)))
                dense_matrix += mask
                dense_matrix = dense_matrix.view(-1, dense_matrix.size(-2), dense_matrix.size(-1))
            
            # get the sparse 4:2 mask
            max_matrix_scores, indices = F.max_pool1d(dense_matrix, kernel_size=4, stride=4, return_indices=True)
            base = torch.empty_like(dense_matrix).fill_(-1e+19)
            base = base.scatter_(2, indices, max_matrix_scores)
            dense_matrix = dense_matrix.scatter(2, indices, -1e+19)
            max_matrix_scores, indices = F.max_pool1d(dense_matrix, kernel_size=4, stride=4, return_indices=True)
            output_matrix_sddmm = base.scatter_(2, indices, max_matrix_scores)

            metadata_reorder_sddmm = None
        else:
            raise NotImplementedError()
    # for bfloat16 tensors
    elif lhs_matrix.dtype == torch.bfloat16:
        output_matrix_sddmm, metadata_reorder_sddmm = sddmm_bf16_ntn(lhs_matrix, rhs_matrix, mask, 1.)
    else:
        raise NotImplementedError()
    
    return output_matrix_sddmm, metadata_reorder_sddmm
