#include <torch/extension.h>  
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cmath>


__global__ void groupNormKernel(
    float* x ,  float* out , int batch_size , int in_channels , int image_size , int group_size
){
    // Groupnorm : (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
    // 32 channels per group  

    namespace cg = cooperative_groups;

    // returns block object for current blockIdx.x
    cg::thread_block block = cg::this_thread_block() ;
    // fine grained control over warps in a block
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block) ;

    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int block_pixels = group_size * image_size ;

    // shared_memory allocation ::   for warp_size = 32  & block_size max = 1024 -> 32*32 = 1024
    __shared__ float smem_sum[32] ;
    __shared__ float smem_sum_sq[32] ;

    // offset x ptr to group initial idx
    x += blockIdx.x * block_pixels ;
    out += blockidx.x * block_pixels ;

    // accumulate sums per thread
    float thread_sum = 0.0f ;
    float thread_sum_sq = 0.0f ;

    // loop through the group/block pixels 
    // mem coalaesced access ::   threads within warp's global mem access  is coalaesced
    // for exm if blockDim.x==512   ||  threadIdx 0  will access : 0 , 512, 1024
    //                              ||  threadIdx 1 will access :  1 , 513, 1025
    for(int i = 0 ; i < block_pixels ; i += blockDim.x ){
        float val = x[i] ;
        thread_sum += val ;
        thread_sum_sq += val * val ;  // square val  
    }

    // warp sum reduce 
    float warp_sum = cg::reduce(warp , thread_sum, cg::plus<float>{} ) ;
    float warp_sum_sq = cg::reduce(warp , thread_sum_sq , cg::plus<float>{} ) ;

    // faith in Independent Thread Scheduling :: small if conditional 
    // thread diverence wont affect as much as writing 32 (altough data safe) race 
    // condition writes 
    if(lane_id == 0 ){
        // thread 0 of warp will store warp sum into shared memory  :: rest skip this
        smem_sum[warp_id] = warp_sum;
        smem_sum_sq[warp_id] = warp_sum_sq;
    }

    // sync the threads across block to acc in shared mem
    __syncthreads() ;


    float mean = 0.0f ;
    float var = 0.0f ;

    // load warp sum from shared mem 
    // first warp will reduce sum this loaded val from smem 
    if(warp_id == 0){
        warp_sum = (lane_id < num_warps) ? smem_sum[lane_id] : 0.0f ;
        warp_sum_sq = (lane_id < num_warps) ? smem_sum_sq[lane_id] : 0.0f ;

        float block_sum =  cg::reduce(warp, warp_sum , cg::plus<float>{} ) ;
        float block_sum_sq =  cg::reduce(warp, warp_sum_sq , cg::plus<float>{} ) ;

        float block_sum /=  block_pixels ;
        float block_sum_sq /= block_pixels ;
        mean = block_sum
        var  = block_sum_sq - mean * mean ;
    }

    int cons_val = rsqrtf(var + 1e-5f )

    // load the final normalized value in same mem coalaesced way 
    for(int i = 0 ; i < block_pixels ; i += blockDim.x){
        float val = cons_val * (x[i] - mean) ;
        out[i] = val ;
    }

}



// launch groupnorm kernel
void groupNorm(
float* x, 
float* out ,
int batch_size ,
int in_channels ,
int height , 
int width 
){
    // Groupnorm : (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
    // 32 channels per group     ::    In_Channels->  [num_groups , channels_per_group]        
    
    int num_group = 32 ;
    int image_size = height * width ;
    int group_size = in_channels / num_group ;

    int grid_size = batch_size * num_group  // num of blocks in grid 
    int block_size =  max_int(min_int(512, group_size * image_size ), 32);  // threads per block

    // launch block for each group 
    groupNormKernel<<<grid_size, block_size>>>(
        x , out , batch_size , in_channels , image_size , group_size
    );

    cudaCheck(cudaGetLastError());

}


/* futher optimization:
- vector load   float4 


*/