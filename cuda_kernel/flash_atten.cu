#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cfloat>
#include <pybind11/pybind11.h>
#include <torch/extension.h> 

constexpr int BLOCK_SIZE = 128;
constexpr int VEC_SIZE = 4; // Use 128-bit vectorized memory operations




template<int TILE_SIZE_Q, int TILE_SIZE_KV>
__global__ void flash_atten_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V,
    float* __restrict__ Out,
    float* __restrict__ LSE,
    int B, int S, int H, int D,
    bool is_causal , float inv_sqrt_d
) {
    // shared memory 
    extern __shared__ char smem_buffer[]; // gives raw smem buffer ptr 
    auto* sQ = reinterpret_cast<float*>(smem_buffer);  // Q_smem_size = TILE_SIZE_Q * D
    auto* sK = sQ + TILE_SIZE_Q * D;        // K_smem_size = TILE_SIZE_KV * D
    auto* sV = sK + TILE_SIZE_KV * D;       // V_smem_size = TILE_SIZE_KV * D

    const int batch_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int block_id = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // vector loading 4 float from D dimension  ||  128-bit vectorized memory operations
    const int d_vec = D / VEC_SIZE ;

    // fine-grained warp control 
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();  // returns block object 
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // initialize l: exp_norm_constant li = rowsum(exp(Si - max_Si))  with zeros for all the row
    // l (TILE_SIZE_Q, 1)   for current block
    // as BLOCK_SIZE = 128 and num_warp = 4 :: each warp_threads will handle one row each out TILE_SIZE_Q
    float l  = 0.0f;       // l for all warp thread reg

    // initialize max_val max_vali = max(max_vali-1 ,rowmax(Si) )   with -ve inf for all the row
    // max_val (TILE_SIZE_Q, 1)  for current block
    // similarily each warp_threads will handle one row each out TILE_SIZE_Q
    float max_val = -FLT_MAX;

    // output accumulator as an array to hold D dimensions per query
    // for D = 128 || D/32 = 4
    float acc[4] = {0.0f}; // for each thread calc multiple value along D(warp_size=32) || D is template param


    // Each warp handles one query row
    const int s_query_block = blockIdx.z * TILE_SIZE_Q;   // S:: TILE_SIZE_Q:TILE_SIZE_Q:TILE_SIZE_Q:.....
    const int s_query = s_query_block + warp_id;          // S:: TILE_SIZE_Q: warp_id:warp_id.. :TILE_SIZE_Q

    // Load Q block from HBM to SMEM
    // Q (B , S , H , D)    || load (TILE_SIZE_Q, D)    
    if(s_query < S && warp_id < TILE_SIZE_Q){   // boundary check 
        const float* q_ptr = Q + (batch_id * S * H + s_query * H + head_id) * D ; // for current warp ,queryrow to load 

        for(int i = lane_id * VEC_SIZE; i < D ; i += 32*VEC_SIZE ){ // warp_threads access pattern-> 0,32,64..... || warp_size = 32
            // cast the float smem & gmem ptr to float4 and then deference the smem ptr to store value
            // sQ + warp_id * D + i : vectorized load by each thread
            *reinterpret_cast<float4*>(&sQ[warp_id * D + i])  = *reinterpret_cast<const float4*>(&q_ptr[i]);  
        }
    }
    __syncthreads();


    // load each of (KV block & compute) in iteration
    for(int tile_kv = 0 ; tile_kv < S ; tile_kv += TILE_SIZE_KV){

        // load KV tile from HBM to SMEM   || K, V both are same dimension block
        // K (B , S , H , D)    || load (TILE_SIZE_KV, D)
        const int kv_end = min(S, tile_kv + TILE_SIZE_KV);  // boundary checking
        const int valid_rows = kv_end - tile_kv;        // to avoid if block divergence inside for loop

        // k_ptr & v_ptr 
        const float* k_ptr = K + (batch_id * S * H + tile_kv * H + head_id) * D;
        const float* v_ptr = V + (batch_id * S * H + tile_kv * H + head_id) * D;

        
        for(int i = tid * VEC_SIZE ; i < valid_rows * D ; i += BLOCK_SIZE * VEC_SIZE) {     // blockDim.x=128  || block_threads access pattern-> 0, 512, 1024..... 
            // (i / D) * D + d  is literally i when flattened from 2D to 1D for SMEM
            //int col = i % D;
            //int row = i / D;       || next Seq row is H * D offset 
            // d_vec = D / vec_size   ||   k_ptr & v_ptr offset :  (i // d_vec) * H * d_vec + i % d_vec)  
            *reinterpret_cast<float4*>(&sK[i])  = *reinterpret_cast<const float4*>(&k_ptr[(i / D) * H * D  + i % D]);   // caution: D should be divisible by VEC_SIZE(4) 
            *reinterpret_cast<float4*>(&sV[i])  = *reinterpret_cast<const float4*>(&v_ptr[(i / D) * H * D  + i % D]);   // or it will cause mem error 
        }
        __syncthreads();


        // not using cuBLAS for GEMM as overhead shadows latency gain from it

        // Process within the KV tile
        if(s_query < S && warp_id < TILE_SIZE_Q){
            float local_max = -FLT_MAX;
            float local_exp_sum = 0.0f;          // alias local_l
            // QK^T : (4, D) * (64, D)^T == (4,64)
            float local_atten[TILE_SIZE_KV] = {0.0f};  // for each row of query output :  (1,64) stored by each warp_threads

            

            // Compute QK^T for keys     || each warp calculate per row of TILE_SIZE_Q
            // (1, D) * (64, D)  == (1, 64)
            for(int i = 0 ; i < valid_rows ; i++) {  // each thread calc one element of TILE_SIZE_Q
                // (1, D) * (1, D)  == (1, 1)    || per thread calc
                float dot = 0.0f ;
                for(int d_idx = lane_id; d_idx < D; d_idx += 32){
                    float q_val = sQ[warp_id * D + d_idx];
                    float k_val = sK[i * D + d_idx];
                    dot += q_val * k_val; // each thread calc its val lane_id 0 : 0, 32, 64,...
                }
                // warpReduce to acc the sum 
                float score = cg::reduce(warp , dot, cg::plus<float>{} ) ;
                // multiply by inv_sqrt_d
                score *= inv_sqrt_d ;

                // Causal masking
                if (is_causal && (s_query < tile_kv + i)) {
                    score = -FLT_MAX;
                }
                
                local_max = fmaxf(local_max, score);   // row max 
                float local_exp = expf(score - max_val);  // softmax* with prev max value

                local_exp_sum += local_exp;       // rowsum local_exp
                local_atten[i] = local_exp;  // softmax* 
            }

            float prev_max = max_val;

            max_val = fmaxf(prev_max, local_max);

            l = (l + local_exp_sum) * expf(prev_max - max_val) ;

            // calculate QK^T . V  == local_atten . V
            // 
            // QK^T (4, 64)   *   V (64, D) == O (4, D)
            for(int i = lane_id ; i < D ; i += 32) { 
                float temp = 0.0f;
                
                for(int j = 0; j < valid_rows; j++){
                    //*reinterpret_cast<float4*>(temp) = reinterpret_cast<float4*>(sV + j * D + i);

                    temp += local_atten[j] * sV[j * D + i];
                }

                acc[warp_id] = (acc[warp_id] + temp) * expf(prev_max - max_val) ; // for one of element along D 
/*
                // for last kv tile 
                if(tile_kv == S-1 ) {
                    acc[warp_id] = acc[warp_id] / l;  
                
                    // store the output 
                    // (B , S , H , D)
                    Out[(batch_id * S * H + s_query * H + head_id) * D + i ] = acc[warp_id];

                    if (lane_id == 0) {
                        LSE[batch_id * S * H + s_query * H + head_id] = logf(l) + max_val;
                    }
                }
*/
            }
        }
        __syncthreads();
    }

    if (s_query < S && warp_id < TILE_SIZE_Q) {
        //
        for (int i = lane_id; i < D; i += 32) {
            Out[(batch_id * S * H + s_query * H + head_id) * D + i] = acc[warp_id] / l;
        }
        if (lane_id == 0) {
            LSE[batch_id * S * H + s_query * H + head_id] = logf(l) + max_val;
        }
    }
}



void launch_attention(
    uintptr_t Q_ptr,
    uintptr_t K_ptr, 
    uintptr_t V_ptr,
    uintptr_t Out_ptr,
    uintptr_t LSE_ptr,
    int B, int S, int H, int D,
    bool is_causal
) {
    const auto* Q = reinterpret_cast<const float*>(Q_ptr); 
    const auto* K = reinterpret_cast<const float*>(K_ptr);
    const auto* V = reinterpret_cast<const float*>(V_ptr);
    auto* Out = reinterpret_cast<float*>(Out_ptr);
    auto* LSE = reinterpret_cast<float*>(LSE_ptr);

    // (B , S , H , D) -> (Batch_size, Seq_len, num_heads, Head_dim)
    // Q , K , V   ->  (B , S , H , D)
    // output initialized with zeros

    constexpr int TILE_SIZE_Q  = 4 ;  // block_size_Q : (4 * D)     ||  num of Q_blocks : S/4 
    constexpr int TILE_SIZE_KV = 16 ; // block_size_KV : (64 * D)    ||  num of KV_blocks S/64
    // output & LSE(log-sum-exp) num_blocks == num_blocks_Q

    dim3 grid(B, H, (S + TILE_SIZE_Q - 1) / TILE_SIZE_Q) ;
    dim3 block(BLOCK_SIZE) ;

    size_t smem_size = (TILE_SIZE_Q * D + 2 * TILE_SIZE_KV * D) * sizeof(float);

    // inv-sqrt-D
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));

    // dynamic shared memory & templated kernel  
    flash_atten_kernel<TILE_SIZE_Q, TILE_SIZE_KV><<<grid, block, smem_size>>>(
        Q, K, V,
        Out, LSE,
        B, S, H, D,
        is_causal , inv_sqrt_d
    );
}