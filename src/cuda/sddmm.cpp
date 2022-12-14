#include <torch/extension.h>
#include <vector>


std::vector<torch::Tensor> sddmm_bf16_ntn_cuda(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::optional<torch::Tensor> mask_,
    const float alpha_);

std::vector<torch::Tensor> sddmm_bf16_ntn(
    torch::Tensor tensor_a_,
    torch::Tensor tensor_b_,
    torch::optional<torch::Tensor> mask_,
    const float alpha_)
{
    return sddmm_bf16_ntn_cuda(tensor_a_, tensor_b_, mask_, alpha_);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("sddmm_bf16_ntn", &sddmm_bf16_ntn, "SDDMM bf16 kernel ntn");
}