#include <cuda.h>
#include <torch/extension.h>
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include <stdio.h>
#include <vector>
#include "cuda_bf16.h"
#include "sddmm/default_sddmma.h"
#include "epilogue/sddmm_epilogue.h"
#include "epilogue/broadcast_mask_epilogue.h"
#include "helper.h"

/// Tiling
using ThreadblockShape_16 = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape_16 = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape_16 = cutlass::gemm::GemmShape<16, 8, 16>;

/// Define MMA & Epilogue
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

/// DefaultConfigurations for float & bf16
using DefaultConfig = cutlass::gemm::device::DefaultGemmConfiguration<
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, float, float, float, float>;

// A struct to switch between different number of stages
template<typename Element_, int Stages>
struct SDDMMConfigure{
    using Element = Element_;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        Element, 128 / cutlass::sizeof_bits<Element>::value, float, Element,
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
    
    using Mma = typename cutlass::gemm::threadblock::DefaultSDDMma<
        Element, cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<Element>::value,
        Element, cutlass::layout::ColumnMajor, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
        float, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        ThreadblockShape_16, WarpShape_16, InstructionShape_16, Stages, cutlass::arch::OpMultiplyAdd>::ThreadblockMma;
    
    using Epilogue = typename cutlass::epilogue::threadblock::DefaultSddmmEpilogue<
        ThreadblockShape_16, WarpShape_16, EpilogueOp, Mma>::Epilogue;
    
    using Prologue = typename cutlass::epilogue::threadblock::DefaultMaskPrologue<
        ThreadblockShape_16, WarpShape_16, EpilogueOp, Mma>::Prologue;
    
    
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
        typename Prologue::SharedStorage prologue;
    };
        
};


template<typename Element, typename _Mma, typename _SharedStorage, typename _Epilogue, typename Prologue, bool Mask>
__global__ void cutlassSddmmKernel_16(
    cutlass::gemm::GemmCoord problem_size,
    cutlass::gemm::GemmCoord grid_tiled_shape,
    typename _Mma::IteratorA::Params params_A,
    Element* __restrict__ ptr_A,
    typename _Mma::IteratorB::Params params_B,
    Element* __restrict__ ptr_B,
    typename _Epilogue::NnzIterator::Params params_D,
    Element* __restrict__ ptr_D,
    typename _Epilogue::IteratorE::Params params_E,
    typename _Epilogue::ElementE* __restrict__ ptr_E,
    typename _Epilogue::OutputOp::Params output_op_,
    int gemm_k_size,
    Element* __restrict__ mask = NULL,
    cutlass::MatrixCoord mask_size = cutlass::MatrixCoord(0, 0))
{
    extern __shared__ int SharedStorageBase[];

    _SharedStorage& shared_storage = *reinterpret_cast<_SharedStorage *>(SharedStorageBase);

    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset=threadblock_swizzle.get_tile_offset(grid_tiled_shape);

    // Early exit if CTA is out of range
    if (grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
        grid_tiled_shape.n() <= threadblock_tile_offset.n())
    {
        return;
    }

    int batch_idx = threadblock_swizzle.get_batch_idx();

    // Compute position within threadblock
    int thread_idx = threadIdx.x;
    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compuled as warp-uniform
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    Prologue prologue;
    
    if (Mask){
        typename Prologue::MaskIterator iterator_M(
            mask, mask_size, thread_idx, batch_idx, cutlass::MatrixCoord{0, threadblock_tile_offset.n() * _Mma::Shape::kN}
        );

        prologue = Prologue(
            shared_storage.prologue,
            thread_idx, warp_idx, lane_idx, iterator_M
        );
    }

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * _Mma::Shape::kM,
        0
    };

    cutlass::MatrixCoord tb_offset_B{
        0, 
        threadblock_tile_offset.n() * _Mma::Shape::kN
    };

    // Problem size
    int problem_size_k = problem_size.k(); //min(problem_size.k(), (threadblock_tile_offset.k() + 1) * gemm_k_size);

    int gemm_k_iterations = (problem_size_k - tb_offset_B.row() + _Mma::Shape::kK - 1) / _Mma::Shape::kK;

    
    // Construct iterators to A, B, and E operands
    typename _Mma::IteratorA iterator_A(
        params_A,
        ptr_A,
        {problem_size.m(), problem_size_k},
        thread_idx,
        tb_offset_A
    );

    iterator_A.add_pointer_offset(batch_idx * problem_size.m() * problem_size.k());

    typename _Mma::IteratorB iterator_B(
        params_B,
        //ref_B.data(),
        ptr_B,
        {problem_size_k, problem_size.n()},
        thread_idx,
        tb_offset_B
    );

    iterator_B.add_pointer_offset(batch_idx * problem_size.n() * problem_size.k());

    //
    //  Main loop
    //

    // Construct thread-scoped matrix multiply
    _Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename _Mma::FragmentC accumulators;

    if (Mask){
        prologue.fill(accumulators);
    } else {
        accumulators.clear();
    }

    if (gemm_k_iterations > 0){
        mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
    }

    //
    //  Epilogue
    //

    typename _Epilogue::OutputOp output_op(output_op_);

    threadblock_tile_offset = threadblock_swizzle.get_tile_offset(grid_tiled_shape);

    // (blockIdx.x * TileM, blockIdx.y * TileN)
    cutlass::MatrixCoord threadblock_offset(
        threadblock_tile_offset.m() * _Mma::Shape::kM,
        threadblock_tile_offset.n() * _Mma::Shape::kN
    );

    cutlass::MatrixCoord tb_offset_E{
        threadblock_tile_offset.m() * _Mma::Shape::kM,
        threadblock_tile_offset.n() * _Mma::Shape::kN
    };

    typename _Epilogue::IteratorE iterator_E(
        params_E,
        ptr_E,
        {problem_size.m(),
        problem_size.n() / _Epilogue::kSparse / _Epilogue::kElementsPerElementE},
        thread_idx,
        tb_offset_E
    );

    iterator_E.add_pointer_offset(batch_idx * problem_size.m() * problem_size.n()/16);

    typename _Epilogue::NnzIterator iterator_D(
        params_D,
        ptr_D,
        problem_size.mn(),
        thread_idx,
        threadblock_offset
    );

    iterator_D.add_pointer_offset(batch_idx * problem_size.m() * problem_size.n()/2);

    _Epilogue epilogue(
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx);

    epilogue(output_op_, iterator_E, iterator_D, accumulators, iterator_E, iterator_D);
}


template<typename Config>
std::vector<torch::Tensor> sddmm_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::optional<torch::Tensor> mask,
    const float alpha_)
{
    const int m = tensor_a.size(-2);
    const int n = tensor_b.size(-2);
    const int k = tensor_b.size(-1);
    
    int batch_size = tensor_a.numel() / m / k;
    int batch_size_b = tensor_b.numel() / n / k;
    assert (batch_size == batch_size_b);

    auto options_val = torch::TensorOptions().dtype(tensor_a.dtype()).device(tensor_b.device());
    auto output_matrix = torch::empty({batch_size, m, n/2}, options_val);

    auto options_meta = torch::TensorOptions().dtype(torch::kInt16).device(tensor_a.device());
    auto metadata = torch::empty({batch_size, m, n/16}, options_meta);

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(m, n, k);

    auto layout_a = cutlass::layout::RowMajor::packed(cutlass::make_Coord(problem_size.m(), problem_size.k()));
    auto layout_b = cutlass::layout::ColumnMajor::packed(problem_size.kn());
    auto layout_e = Config::Epilogue::LayoutE::packed(cutlass::make_Coord(problem_size.m(), problem_size.k() / 16));
    auto layout_d = cutlass::layout::RowMajor::packed(cutlass::make_Coord(problem_size.m(), problem_size.n() / 2));

    typename Config::Element alpha = typename Config::Element(alpha_);
    typename Config::Element beta = typename Config::Element(0.0);
    
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        problem_size,
        {ThreadblockShape_16::kM, ThreadblockShape_16::kN, ThreadblockShape_16::kK},
        batch_size
    );

    dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);
    dim3 block(Config::Mma::WarpCount::kCount * 32, 1, 1);

    int smem_size = int(sizeof(typename Config::SharedStorage));
    int gemm_k_size = ((problem_size.k() + Config::Mma::Shape::kK - 1) / Config::Mma::Shape::kK) * Config::Mma::Shape::kK;


    if (mask.has_value()){
        cudaFuncSetAttribute(cutlassSddmmKernel_16<typename Config::Element, typename Config::Mma, typename Config::SharedStorage, typename Config::Epilogue, typename Config::Prologue, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        cudaFuncSetAttribute(cutlassSddmmKernel_16<typename Config::Element, typename Config::Mma, typename Config::SharedStorage, typename Config::Epilogue, typename Config::Prologue, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cutlass::MatrixCoord mask_shape{cutlass::MatrixCoord::Index(batch_size / (mask.value().numel()/n)), cutlass::MatrixCoord::Index(n)};
        cutlassSddmmKernel_16<
        typename Config::Element, typename Config::Mma, 
        typename Config::SharedStorage, typename Config::Epilogue, typename Config::Prologue, true><<<grid, block, smem_size>>>(
            problem_size, grid_tiled_shape, 
            layout_a, (typename Config::Element*)tensor_a.data_ptr(),
            layout_b, (typename Config::Element*)tensor_b.data_ptr(),
            layout_d, (typename Config::Element*)output_matrix.data_ptr(),
            layout_e, (uint16_t*)metadata.data_ptr(), {alpha, beta}, gemm_k_size,
            (typename Config::Element*)mask.value().data_ptr(), mask_shape
        );
    } else {
        cudaFuncSetAttribute(cutlassSddmmKernel_16<typename Config::Element, typename Config::Mma, typename Config::SharedStorage, typename Config::Epilogue, typename Config::Prologue, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        cudaFuncSetAttribute(cutlassSddmmKernel_16<typename Config::Element, typename Config::Mma, typename Config::SharedStorage, typename Config::Epilogue, typename Config::Prologue, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

        cutlassSddmmKernel_16<
        typename Config::Element, typename Config::Mma, 
        typename Config::SharedStorage, typename Config::Epilogue, typename Config::Prologue, false><<<grid, block, smem_size>>>(
            problem_size, grid_tiled_shape, 
            layout_a, (typename Config::Element*)tensor_a.data_ptr(),
            layout_b, (typename Config::Element*)tensor_b.data_ptr(),
            layout_d, (typename Config::Element*)output_matrix.data_ptr(),
            layout_e, (uint16_t*)metadata.data_ptr(), {alpha, beta}, gemm_k_size);
    }

    
    
    return {output_matrix, metadata};
}

std::vector<torch::Tensor> sddmm_bf16_ntn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::optional<torch::Tensor> mask, const float alpha)
{
    const int k = tensor_b.size(-1);

    if (k == 64){
        using Config = SDDMMConfigure<cutlass::bfloat16_t, 1>;
        return sddmm_cuda<Config>(tensor_a, tensor_b, mask, alpha);
    } else if (k == 128){
        using Config = SDDMMConfigure<cutlass::bfloat16_t, 2>;
        return sddmm_cuda<Config>(tensor_a, tensor_b, mask, alpha);
    } else {
        using Config = SDDMMConfigure<cutlass::bfloat16_t, 3>;
        return sddmm_cuda<Config>(tensor_a, tensor_b, mask, alpha);
    }
}


std::vector<torch::Tensor> sddmm_f16_ntn_cuda(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::optional<torch::Tensor> mask, const float alpha)
{
    const int k = tensor_b.size(-1);
    if (k == 64){
        using Config = SDDMMConfigure<cutlass::half_t, 1>;
        return sddmm_cuda<Config>(tensor_a, tensor_b, mask, alpha);
    } else if (k == 128){
        using Config = SDDMMConfigure<cutlass::half_t, 2>;
        return sddmm_cuda<Config>(tensor_a, tensor_b, mask, alpha);
    } else {
        using Config = SDDMMConfigure<cutlass::half_t, 3>;
        return sddmm_cuda<Config>(tensor_a, tensor_b, mask, alpha);
    }
}