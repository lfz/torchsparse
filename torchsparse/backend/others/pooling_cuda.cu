#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>
#include <vector>

#include "pooling_cuda.h"

namespace {

template <typename scalar_t>
__global__ void sparse_maxpool_forward_kernel(
    const scalar_t* __restrict__ feats,
    const int* __restrict__ out_in_map,
    scalar_t* __restrict__ output,
    int* __restrict__ argmax,
    int num_out,
    int kernel_volume,
    int channels) {
  int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_out * channels;
  if (linear_idx >= total) {
    return;
  }

  int out_idx = linear_idx / channels;
  int channel = linear_idx - out_idx * channels;

  float max_val = -std::numeric_limits<float>::infinity();
  int max_index = -1;

  for (int k = 0; k < kernel_volume; ++k) {
    int in_idx = out_in_map[out_idx * kernel_volume + k];
    if (in_idx < 0) {
      continue;
    }
    float val = static_cast<float>(feats[in_idx * channels + channel]);
    if (val > max_val) {
      max_val = val;
      max_index = in_idx;
    }
  }

  output[linear_idx] =
      max_index >= 0 ? static_cast<scalar_t>(max_val) : static_cast<scalar_t>(0);
  argmax[linear_idx] = max_index;
}

template <typename scalar_t>
__global__ void sparse_maxpool_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const int* __restrict__ argmax,
    scalar_t* __restrict__ grad_input,
    int num_out,
    int channels) {
  int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_out * channels;
  if (linear_idx >= total) {
    return;
  }

  int input_index = argmax[linear_idx];
  if (input_index < 0) {
    return;
  }
  int channel = linear_idx - (linear_idx / channels) * channels;
  scalar_t grad_val = grad_output[linear_idx];
  gpuAtomicAdd(
      grad_input + input_index * channels + channel, grad_val);
}

template <typename scalar_t>
__global__ void sparse_avgpool_forward_kernel(
    const scalar_t* __restrict__ feats,
    const int* __restrict__ out_in_map,
    scalar_t* __restrict__ output,
    int* __restrict__ counts,
    int num_out,
    int kernel_volume,
    int channels) {
  int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_out * channels;
  if (linear_idx >= total) {
    return;
  }

  int out_idx = linear_idx / channels;
  int channel = linear_idx - out_idx * channels;

  float sum_val = 0.f;
  int valid_count = 0;

  for (int k = 0; k < kernel_volume; ++k) {
    int in_idx = out_in_map[out_idx * kernel_volume + k];
    if (in_idx < 0) {
      continue;
    }
    sum_val += static_cast<float>(feats[in_idx * channels + channel]);
    ++valid_count;
  }

  float avg = (valid_count > 0) ? (sum_val / static_cast<float>(valid_count)) : 0.f;
  output[linear_idx] = static_cast<scalar_t>(avg);

  if (channel == 0) {
    counts[out_idx] = valid_count;
  }
}

template <typename scalar_t>
__global__ void sparse_avgpool_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const int* __restrict__ out_in_map,
    const int* __restrict__ counts,
    scalar_t* __restrict__ grad_input,
    int num_out,
    int kernel_volume,
    int channels) {
  int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_out * channels;
  if (linear_idx >= total) {
    return;
  }

  int out_idx = linear_idx / channels;
  int channel = linear_idx - out_idx * channels;

  int valid_count = counts[out_idx];
  if (valid_count <= 0) {
    return;
  }

  scalar_t grad_val =
      grad_output[linear_idx] / static_cast<scalar_t>(valid_count);

  for (int k = 0; k < kernel_volume; ++k) {
    int in_idx = out_in_map[out_idx * kernel_volume + k];
    if (in_idx < 0) {
      continue;
    }
    gpuAtomicAdd(
        grad_input + in_idx * channels + channel, grad_val);
  }
}

std::tuple<int, int, int> get_pool_shape(
    const at::Tensor& feats,
    const at::Tensor& out_in_map) {
  int num_out = out_in_map.size(0);
  int kernel_volume = out_in_map.size(1);
  int channels = feats.size(1);
  return std::make_tuple(num_out, kernel_volume, channels);
}

}  // namespace

std::vector<at::Tensor> sparse_maxpool_forward_cuda(
    at::Tensor feats,
    at::Tensor out_in_map) {
  TORCH_CHECK(
      feats.is_cuda(), "sparse_maxpool_forward_cuda: feats must be a CUDA tensor.");
  TORCH_CHECK(
      out_in_map.is_cuda(),
      "sparse_maxpool_forward_cuda: out_in_map must be a CUDA tensor.");

  feats = feats.contiguous();
  out_in_map = out_in_map.contiguous();

  auto [num_out, kernel_volume, channels] =
      get_pool_shape(feats, out_in_map);

  auto feat_options = feats.options();
  auto idx_options =
      out_in_map.options().dtype(at::kInt);

  at::Tensor output = at::zeros({num_out, channels}, feat_options);
  at::Tensor argmax = at::full({num_out, channels}, -1, idx_options);

  int total = num_out * channels;
  if (total > 0) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        feats.scalar_type(),
        "sparse_maxpool_forward_kernel",
        [&] {
          sparse_maxpool_forward_kernel<scalar_t><<<blocks, threads>>>(
              feats.data_ptr<scalar_t>(),
              out_in_map.data_ptr<int>(),
              output.data_ptr<scalar_t>(),
              argmax.data_ptr<int>(),
              num_out,
              kernel_volume,
              channels);
        });
    AT_CUDA_CHECK(cudaGetLastError());
  }

  return {output, argmax};
}

at::Tensor sparse_maxpool_backward_cuda(
    at::Tensor grad_output,
    at::Tensor argmax,
    int input_size) {
  TORCH_CHECK(
      grad_output.is_cuda(),
      "sparse_maxpool_backward_cuda: grad_output must be CUDA tensor.");
  TORCH_CHECK(
      argmax.is_cuda(),
      "sparse_maxpool_backward_cuda: argmax must be CUDA tensor.");

  grad_output = grad_output.contiguous();
  argmax = argmax.contiguous();

  int channels = grad_output.size(1);
  at::Tensor grad_input =
      at::zeros({input_size, channels}, grad_output.options());

  int num_out = grad_output.size(0);
  int total = num_out * channels;
  if (total > 0) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_output.scalar_type(),
        "sparse_maxpool_backward_kernel",
        [&] {
          sparse_maxpool_backward_kernel<scalar_t><<<blocks, threads>>>(
              grad_output.data_ptr<scalar_t>(),
              argmax.data_ptr<int>(),
              grad_input.data_ptr<scalar_t>(),
              num_out,
              channels);
        });
    AT_CUDA_CHECK(cudaGetLastError());
  }

  return grad_input;
}

std::vector<at::Tensor> sparse_avgpool_forward_cuda(
    at::Tensor feats,
    at::Tensor out_in_map) {
  TORCH_CHECK(
      feats.is_cuda(), "sparse_avgpool_forward_cuda: feats must be a CUDA tensor.");
  TORCH_CHECK(
      out_in_map.is_cuda(),
      "sparse_avgpool_forward_cuda: out_in_map must be a CUDA tensor.");

  feats = feats.contiguous();
  out_in_map = out_in_map.contiguous();

  auto [num_out, kernel_volume, channels] =
      get_pool_shape(feats, out_in_map);

  auto feat_options = feats.options();
  auto idx_options =
      out_in_map.options().dtype(at::kInt);

  at::Tensor output = at::zeros({num_out, channels}, feat_options);
  at::Tensor counts = at::zeros({num_out}, idx_options);

  int total = num_out * channels;
  if (total > 0) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        feats.scalar_type(),
        "sparse_avgpool_forward_kernel",
        [&] {
          sparse_avgpool_forward_kernel<scalar_t><<<blocks, threads>>>(
              feats.data_ptr<scalar_t>(),
              out_in_map.data_ptr<int>(),
              output.data_ptr<scalar_t>(),
              counts.data_ptr<int>(),
              num_out,
              kernel_volume,
              channels);
        });
    AT_CUDA_CHECK(cudaGetLastError());
  }

  return {output, counts};
}

at::Tensor sparse_avgpool_backward_cuda(
    at::Tensor grad_output,
    at::Tensor out_in_map,
    at::Tensor counts,
    int input_size) {
  TORCH_CHECK(
      grad_output.is_cuda(),
      "sparse_avgpool_backward_cuda: grad_output must be CUDA tensor.");
  TORCH_CHECK(
      out_in_map.is_cuda(),
      "sparse_avgpool_backward_cuda: out_in_map must be CUDA tensor.");
  TORCH_CHECK(
      counts.is_cuda(),
      "sparse_avgpool_backward_cuda: counts must be CUDA tensor.");

  grad_output = grad_output.contiguous();
  out_in_map = out_in_map.contiguous();
  counts = counts.contiguous();

  int channels = grad_output.size(1);
  at::Tensor grad_input =
      at::zeros({input_size, channels}, grad_output.options());

  int num_out = grad_output.size(0);
  auto [_, kernel_volume, __] = get_pool_shape(grad_output, out_in_map);

  int total = num_out * channels;
  if (total > 0) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_output.scalar_type(),
        "sparse_avgpool_backward_kernel",
        [&] {
          sparse_avgpool_backward_kernel<scalar_t><<<blocks, threads>>>(
              grad_output.data_ptr<scalar_t>(),
              out_in_map.data_ptr<int>(),
              counts.data_ptr<int>(),
              grad_input.data_ptr<scalar_t>(),
              num_out,
              kernel_volume,
              channels);
        });
    AT_CUDA_CHECK(cudaGetLastError());
  }

  return grad_input;
}
