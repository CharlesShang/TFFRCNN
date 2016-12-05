#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_PSROIPOOLING_OP_GPU_H_
#define TENSORFLOW_USER_OPS_PSROIPOOLING_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Run the forward pass of max pooling, optionally writing the argmax indices to
// the mask array, if it is not nullptr. If mask is passed in as nullptr, the
// argmax indices are not written.
bool PSROIPoolForwardLauncher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width, const float* bottom_rois,
    const int output_dim, const int group_size, float* top_data, int* mapping_channel, const Eigen::GpuDevice& d);

bool PSROIPoolBackwardLauncher(const float* top_diff, const int* mapping_channel, const int num_rois, const float spatial_scale,
                               const int channels, const int height, const int width, const int pooled_height, const int pooled_width,
                               const int output_dim, float* bottom_diff, const float* bottom_rois, const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_GPU_H_