#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include "psroi_pooling_op_gpu.h"
#include "cuda_kernel_helper.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

using std::max;
using std::min;


using namespace tensorflow;

template <typename Dtype>
  __global__ void PSROIPoolingForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,
    const int output_dim,
    const int group_size,
    Dtype* top_data,
    int* mapping_channel) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
      Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
      Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
      Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom 
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      int hstart = floor(static_cast<Dtype>(ph) * bin_size_h
                          + roi_start_h);
      int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
                          + roi_start_w);
      int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
                        + roi_start_h);
      int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
                        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0),width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      int gw = pw;
      int gh = ph;
      int c = (ctop*group_size + gh)*group_size + gw;

      bottom_data += (roi_batch_ind * channels + c) * height * width;
      Dtype out_sum = 0;
      for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w){
          int bottom_index = h*width + w;
          out_sum += bottom_data[bottom_index];
        }
      }

      Dtype bin_area = (hend - hstart)*(wend - wstart);
      top_data[index] = is_empty? 0. : out_sum / bin_area ;
      mapping_channel[index] = c;
    }
}


bool PSROIPoolForwardLauncher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width, const float* bottom_rois,
    const int output_dim, const int group_size, float* top_data, int* mapping_channel, const Eigen::GpuDevice& d) 
{
  const int kThreadsPerBlock = 1024;
  const int output_size = num_rois * pooled_height * pooled_width * output_dim;
  cudaError_t err;

  PSROIPoolingForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(output_size, bottom_data, spatial_scale, channels, height, width, 
                                                          pooled_height, pooled_width, bottom_rois, output_dim, group_size, 
                                                          top_data, mapping_channel);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}

  template <typename Dtype>
  __global__ void PSROIPoolingBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim, 
    Dtype* bottom_diff,
    const Dtype* bottom_rois) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
      Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
      Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
      Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom 
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      int hstart = floor(static_cast<Dtype>(ph)* bin_size_h
        + roi_start_h);
      int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
        + roi_start_w);
      int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
        + roi_start_h);
      int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      int c = mapping_channel[index];
      Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
      Dtype bin_area = (hend - hstart)*(wend - wstart);
      Dtype diff_val = is_empty ? 0. : top_diff[index] / bin_area;
      
      for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w){
          int bottom_index = h*width + w;
          //caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
          if (c != -1) {
              CudaAtomicAdd(offset_bottom_diff + bottom_index, diff_val);
            }
        }
      }
    }
}

bool PSROIPoolBackwardLauncher(const float* top_diff, const int* mapping_channel, const int num_rois, const float spatial_scale,
                               const int channels, const int height, const int width, const int pooled_height, const int pooled_width,
                               const int output_dim, float* bottom_diff, const float* bottom_rois, const Eigen::GpuDevice& d) 
{
  const int kThreadsPerBlock = 1024;
  const int output_size = num_rois * pooled_height * pooled_width * output_dim;
  const int bottom_size = 1 * height * width * channels;
  cudaError_t err;

  SetZero<<<(bottom_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, d.stream()>>>(bottom_size, bottom_diff);
  
  PSROIPoolingBackwardAtomic<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(output_size, top_diff, mapping_channel, num_rois, spatial_scale, channels,
                                                          height, width, pooled_height, pooled_width, output_dim, bottom_diff, bottom_rois);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}

#endif  // GOOGLE_CUDA