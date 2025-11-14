#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> sparse_maxpool_forward_cuda(
    at::Tensor feats, at::Tensor out_in_map);
at::Tensor sparse_maxpool_backward_cuda(
    at::Tensor grad_output, at::Tensor argmax, int input_size);

std::vector<at::Tensor> sparse_avgpool_forward_cuda(
    at::Tensor feats, at::Tensor out_in_map);
at::Tensor sparse_avgpool_backward_cuda(
    at::Tensor grad_output,
    at::Tensor out_in_map,
    at::Tensor counts,
    int input_size);
