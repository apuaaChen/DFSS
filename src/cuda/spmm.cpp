#include <torch/extension.h>
#include <vector>


torch::Tensor spmmv2_bf16_nnn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha);

torch::Tensor spmmv2_bf16_nnn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::Tensor tensor_e_,
    const float alpha)
{
    return spmmv2_bf16_nnn_cuda(tensor_a_, tensor_b_, tensor_e_, alpha);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("spmm_bf16_nnn", &spmm_bf16_nnn, "Cutlass SpMM bf16 kernel nnn");
}