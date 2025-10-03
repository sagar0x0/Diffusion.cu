#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
// vector loading

// CUDA error checking
inline void cuda_check(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

__global__ void groupNormKernel(float *x, float *out, int batch_size,
                                int in_channels, int image_size,
                                int group_size) {
  // Groupnorm : (Batch_Size, In_Channels, Height, Width) -> (Batch_Size,
  // In_Channels, Height, Width) 32 channels per group

  namespace cg = cooperative_groups;

  // returns block object for current blockIdx.x
  cg::thread_block block = cg::this_thread_block();
  // fine grained control over warps in a block
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int block_pixels = group_size * image_size;

  // shared_memory allocation ::   for warp_size = 32  & block_size max = 1024
  // as 32 *32 = 1024
  __shared__ float smem_sum[32];
  __shared__ float smem_sum_sq[32];

  // shared mean , inv_std_val
  __shared__ float S_mean;
  __shared__ float S_inv_std; // rsqrtf(var + 1e-5f)

  // offset x ptr to group initial idx
  x += blockIdx.x * block_pixels;
  out += blockIdx.x * block_pixels;

  // accumulate sums per thread
  float thread_sum = 0.0;
  float thread_sum_sq = 0.0;

  // loop through the group/block pixels       ||  float4 vector load
  // mem coalaesced access :: threads within warp's global mem access  is
  // coalaesced for exm if blockDim.x==512   
  //  threadIdx 0  will access : 0 ,512, 1024  ||  threadIdx 1 will access :  1 , 513, 1025
  //  caution: block_pixels should be 16 bytes aligned
  for (int i = threadIdx.x * 4; i < block_pixels;i += blockDim.x * 4) { 
    float4 vec = *reinterpret_cast<float4 *>(&x[i]);
    thread_sum += vec.x + vec.y + vec.z + vec.w;
    thread_sum_sq += (vec.x * vec.x) + (vec.y * vec.y) +
                     (vec.z * vec.z) +(vec.w * vec.w);  // square val
  }

  // warp sum reduce
  float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>{});
  float warp_sum_sq = cg::reduce(warp, thread_sum_sq, cg::plus<float>{});

  // faith in Independent Thread Scheduling :: small if conditional
  // thread diverence wont affect as much as writing 32 (altough data safe) race
  // condition writes
  if (lane_id == 0) {
    // thread 0 of warp will store warp sum into shared memory  :: rest skip
    // this
    smem_sum[warp_id] = warp_sum;
    smem_sum_sq[warp_id] = warp_sum_sq;
  }

  // sync the threads across block to acc in shared mem
  __syncthreads();

  // load warp sum from shared mem
  // first warp will reduce sum this loaded val from smem
  if (warp_id == 0) {
    warp_sum = (lane_id < num_warps) ? smem_sum[lane_id] : 0.0f;
    warp_sum_sq = (lane_id < num_warps) ? smem_sum_sq[lane_id] : 0.0f;

    float block_sum = cg::reduce(warp, warp_sum, cg::plus<float>{});
    float block_sum_sq = cg::reduce(warp, warp_sum_sq, cg::plus<float>{});

    // store the mean adn var to shared mem
    if (threadIdx.x == 0) {
      float mean = block_sum / block_pixels;
      float var = (block_sum_sq / block_pixels) - (mean * mean);

      S_mean = mean;
      S_inv_std = rsqrtf(var + 1e-5f);
    }
  }

  __syncthreads();

  float mean = S_mean;
  float inv_std_val = S_inv_std;

  // load the final normalized value in same mem coalaesced way
  // vectorized store and load
  // caution: needs to be 16 byte aligned
  for (int i = threadIdx.x * 4; i < block_pixels; i += blockDim.x * 4) { 
    float4 vec_in = *reinterpret_cast<const float4 *>(&x[i]); // vec load form smem

    float4 vec_out;
    vec_out.x = (vec_in.x - mean) * inv_std_val;
    vec_out.y = (vec_in.y - mean) * inv_std_val;
    vec_out.z = (vec_in.z - mean) * inv_std_val;
    vec_out.w = (vec_in.w - mean) * inv_std_val;

    *reinterpret_cast<float4 *>(&out[i]) = vec_out; // vec store to gmem
  }
}

// launch groupnorm kernel
void groupNorm(uintptr_t x_ptr, uintptr_t out_ptr, int batch_size,
               int in_channels, int height, int width, int num_groups) {
  auto *x = reinterpret_cast<float *>(x_ptr);
  auto *out = reinterpret_cast<float *>(out_ptr);
  // Groupnorm : (Batch_Size, In_Channels, Height, Width) -> (Batch_Size,
  // In_Channels, Height, Width) 32 channels per group     ::    In_Channels->
  // [num_groups , channels_per_group]

  // int num_group = 32;  || take from kernel call arg
  int num_group = num_groups;
  int image_size = height * width;
  int group_size = in_channels / num_group;

  int grid_size = batch_size * num_group; // num of blocks in grid
  int block_size =
      max(min(512, group_size * image_size), 32); // threads per block

  // launch block for each group
  groupNormKernel<<<grid_size, block_size>>>(x, out, batch_size, in_channels,
                                             image_size, group_size);

  cudaCheck(cudaGetLastError());
}


PYBIND11_MODULE(groupNorm_kernel, m) {
    m.def("groupNorm", &groupNorm, "Cuda groupNorm kernel");
}