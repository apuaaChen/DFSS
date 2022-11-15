"""
Entrypoint fo SpMM kernels. It is implemented based on the threadblock level APIs in CUTLASS
"""
import torch
from dfss.spmm import spmmv2_bf16_nnn


def spmm(nonzeros: torch.Tensor, metadata: torch.Tensor, rhs_matrix: torch.Tensor) -> torch.Tensor:
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

    #########################
    # Check input dimension #
    #########################

    if nonzeros.dim() != 2 and nonzeros.dim() != 3:
        raise ValueError("expected 2D or 3D nonzeros (got {}D input)".format(nonzeros.dim()))
    
    if metadata.dim() != 2 and metadata.dim() != 3:
        raise ValueError("expected 2D or 3D metadata (got {}D input)".format(metadata.dim()))

    if rhs_matrix.dim() != 2 and rhs_matrix.dim() != 3:
        raise ValueError("expected 2D or 3D rhs_matrix (got {}D input)".format(rhs_matrix.dim()))
    
    if nonzeros.dim() != metadata.dim() or nonzeros.dim() != rhs_matrix.dim() or metadata.dim() != rhs_matrix.dim():
        raise ValueError("the two input matrices must have the "
                         "same dimension (got {}D nonzeros, {}D metadata, and {}D rhs_matrix)".format(
                             nonzeros.dim(), metadata.dim(), rhs_matrix.dim()))
    
    if nonzeros.dim() == 3 and (
        nonzeros.size(0) != metadata.size(0) or nonzeros.size(0) != rhs_matrix.size(0) or metadata.size(0) != rhs_matrix.size(0)):
        raise ValueError("the batch size should be the same (got B={} in nonzeros, B={} in metadata, and B={} in rhs_matrix)".format(
            nonzeros.size(0), metadata.size(0), rhs_matrix.size(0)))
    
    if nonzeros.size(-1) * 2 != rhs_matrix.size(-2):
        raise ValueError("the reduction dim N of nonzeros should be half of rhs_matrix (got K={} in lhs_matrix and K={} in rhs_matrix".format(nonzeros.size(-1), rhs_matrix.size(-2)))
    
    #########################
    # Check input data type #
    #########################

    if nonzeros.dtype != torch.float32 and nonzeros.dtype != torch.bfloat16:
        raise ValueError("the nonzero should be in torch.float32 or torch.bfloat16 (got {})".format(nonzeros.dtype))
    
    if metadata.dtype != torch.int16:
        raise ValueError("the metadata should be in torch.int16 (got {})".format(metadata.dtype))
    
    if rhs_matrix.dtype != torch.float32 and rhs_matrix.dtype != torch.bfloat16:
        raise ValueError("the rhs_matrix should be in torch.float32 or torch.bfloat16 (got {})".format(rhs_matrix.dtype))

    if not nonzeros.is_cuda:
        raise ValueError("the nonzeros should be on GPU (got CPU)")
    
    if not metadata.is_cuda:
        raise ValueError("the metadata should be on GPU (got CPU)")
    
    if not rhs_matrix.is_cuda:
        raise ValueError("the rhs_matrix should be on GPU (got CPU)")
    
    ################################
    # launch the extended function #
    ################################

    # for float32 tensors
    if nonzeros.dtype == torch.float32:
        raise NotImplementedError()
    # for bfloat16 tensor
    else:
        output_matrix = spmmv2_bf16_nnn(nonzeros, rhs_matrix, metadata)
    
    return output_matrix
        