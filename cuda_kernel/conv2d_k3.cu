#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cmath>
#include <pybind11/pybind11.h>

// CUDA error checking
inline void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))


#define TILE_WIDTH 16
#define TH 4 
#define TW 4
#define cin_chunk 16


__global__ void conv2d_k3_kernel(
    const float* x , const float* w , float* out ,
    const int Batch_size, const int C_in, const int C_out, const int Height, const int Width,
    int num_cin
) {
    // x :: (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
    // weight filter :: (out_channels, in_channels, 3, 3)      || kerenel_height == kernel_width == 3

    int block_row = blockIdx.y ;
    int block_col = blockIdx.x ;

    int batch_idx = blockIdx.y / (C_out * num_cin) ;
    // int cout_idx = batch_idx * C_out * num_cin + blockIdx.y / num_cin ;
    int cin_idx = blockIdx.y % num_cin ;

    int c_out_idx = (blockIdx.y / num_cin) % C_out ;   // 


    // per threadBlock we will take 16 input_channels 
    // calculate the working threadBlock
    int block_idx = blockIdx.y / 16 ;
    
    __shared__ float x_smem[16 * (TILE_WIDTH + 2)  * (TILE_WIDTH + 2)  ] ;   // input tile needed to calc output tile
    __shared__ float w_smem[16 * 3 * 3 ] ;   // weight filter/kernel 

    // x :: (Batch_Size, In_Channels, Height, Width)
    int x_off = batch_idx * C_in * Height * Width + 
                cin_idx * 16 * Height * Width;

    int thread_row = threadIdx.x / (TILE_WIDTH + 2);
    int thread_col = threadIdx.x % (TILE_WIDTH + 2);

    // Tile positioning     ||   16*16  tile 
    int tiles_per_row = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
    int tile_row = blockIdx.x / tiles_per_row;
    int tile_col = blockIdx.x % tiles_per_row;
    int output_tile_row = tile_row * TILE_WIDTH;
    int output_tile_col = tile_col * TILE_WIDTH;

    for (int i = 0; i < 16; i++) {
        const int x_mem_off = i * (TILE_WIDTH + 2) * (TILE_WIDTH + 2);
        const int current_x_off = x_off + i * Height * Width;

        const int in_row = output_tile_row + (thread_row - 1) ;
        const int in_col = output_tile_col + (thread_col - 1)  ;
        const int smem_idx = x_mem_off + thread_row * (TILE_WIDTH + 2) + thread_col;

        if (in_row >= 0 && in_row < Height && in_col >= 0 && in_col < Width) {
            x_smem[smem_idx] = x[current_x_off + in_row * Width + in_col];
        } else {
            x_smem[smem_idx] = 0.0f; // zero-padding
        }
    }
    __syncthreads();


    // load filter weight
    // weight filter :: (out_channels, in_channels, 3, 3)
    int filter_off = (c_out_idx * C_in * 9) + (cin_idx * 16 * 9)  ;
    
    int filter_idx = threadIdx.x ;

    if(filter_idx < 144 ) {  //  16 * 3 * 3  == 144
        w_smem[filter_idx] = w[filter_off + filter_idx] ;
    }
    __syncthreads() ;

    // ------------------ shared_mem loaded -------------------------------



    float c_out_temp =  0.0f ;   // holds temp acc value for each c_in layer

    // conv calc for output channel taking 16 C_in and (TILE_WIDTH * TILE_WIDTH) at once
    if(thread_col <= TILE_WIDTH && thread_row <= TILE_WIDTH && thread_col >= 1 && thread_row >= 1){
        for(int i = 0 ; i < 16 ; i++){
            float matmul_temp = 0.0f ;
            int x_smem_off = i * (TILE_WIDTH + 2) * (TILE_WIDTH + 2) ;
                            
            int filter_off = i * 3* 3 ;

            for(int j = 0 ; j < 3 ; j++){
                for(int k = 0 ; k < 3 ; k++){
                    int current_smem_row = thread_row - 1 + j;
                    int current_smem_col = thread_col - 1 + k;
                    matmul_temp += x_smem[x_smem_off + current_smem_row*(TILE_WIDTH + 2)  + current_smem_col] *
                                 w_smem[filter_off + j*3 + k] ;
                }
            }
            c_out_temp +=  matmul_temp ;
        }       
    }


    // store c_out_temp to C_out 
    // out :: (Batch_Size, Out_Channels, Height, Width)

    int out_off = (batch_idx * C_out * Height * Width) + 
                (c_out_idx * Height * Width) + 
                output_tile_row * Width +
                output_tile_col  ;
    
    if(thread_col <= TILE_WIDTH && thread_row <= TILE_WIDTH && thread_col >= 1 && thread_row >= 1){
        int out_idx = (thread_row - 1) * Width + (thread_col - 1);
        atomicAdd(&out[out_off + out_idx], c_out_temp);
    }
    

}


void conv2d_kernel(
    uintptr_t x_ptr , 
    uintptr_t w_ptr , 
    uintptr_t out_ptr ,
    const int Batch_size, const int C_in, const int C_out, const int Height, const int Width
) {

    const auto* x = reinterpret_cast<const float*>(x_ptr);
    const auto* w = reinterpret_cast<const float*>(w_ptr);
    auto* out = reinterpret_cast<float*>(out_ptr);



    cudaCheck(cudaMemset(out, 0, Batch_size * C_out * Height * Width * sizeof(float))) ;

    // (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)

    int thread_block =  (TILE_WIDTH + 2) * (TILE_WIDTH + 2) ;   // min height and width needed == 16 

    int num_blocks = (Height * Width) / (TILE_WIDTH * TILE_WIDTH) ;   // num_blocks lunched == num of output tiled blocks
    
    int num_cin = C_in / 16 ;    // num_cin = C_in / 16   || as 1 threadBlock takes 16 cin at a time   

    dim3 blockDim(thread_block) ;
    dim3 gridDim(num_blocks , Batch_size * C_out * num_cin ) ;

    conv2d_k3_kernel<<<gridDim , blockDim >>>(
        x, w, out, 
        Batch_size, C_in, C_out, Height, Width,
        num_cin
    ) ;

    cudaCheck(cudaGetLastError()) ;

}

PYBIND11_MODULE(conv2dkernel, m) {
  m.def("Conv2d_cuda_kernel", &conv2d_kernel, "Cuda Conv2d kernel");
}