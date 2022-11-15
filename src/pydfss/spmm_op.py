"""
Entrypoint fo SpMM kernels. It is implemented based on the threadblock level APIs in CUTLASS
"""
import torch
from dfss.spmm import spmm_bf16_nnn


def spmm(nonzeros: torch.Tensor, metadata: torch.Tensor, rhs_matrix: torch.Tensor, training=False) -> torch.Tensor:
    r""" The SpMM kernel works on the 50% sparsity on Ampere. 
    It performs :math:`AxB=C`, :math:`A` is under the 50% structured sparse encoding,
    :math:`B` and :math:`C` are row-major matrices.

    Args:
        nonzeros: the nonzero values in :math:`A` of size `(B, M, N/2)` or `(M, N/2)`.
        metadata: the meta data that encodes the 50% sparse matrix
        rhs_matrix: :math:`B` of size `(B, N, K)` or `(N, K)`.
    
    Returns:
        output_matrix: :math:`C` of size `(B, M, K)` or `(M, K)`
    
    Example:
        >>> import torch
        >>> import dspattn
        >>> dense_matrix = torch.randn(size=(8, 4096, 4096), dtype=torch.bfloat16, device='cuda')
        >>> nonzeros, metadata = dspattn.dense2sparse(dense_matrix)
        >>> rhs_matrix = torch.randn(size=(8, 4096, 64), dtype=torch.bfloat16, device='cuda')
        >>> output_matrix = dspattn.spmm(nonzeros, metadata, rhs_matrix)
    """
    
    ################################
    # launch the extended function #
    ################################

    # for float32 tensors
    if nonzeros.dtype == torch.float32:
        if training:
            output_matrix = torch.matmul(nonzeros, rhs_matrix)
        else:
            raise NotImplementedError()
    # for bfloat16 tensor
    else:
        output_matrix = spmm_bf16_nnn(nonzeros, rhs_matrix, metadata, 1.)
    
    return output_matrix
        