# Amazon SDE Intern Interview Prep — Diffusion.cu

---

## 1. Project Summary (Architectural Overview)

Diffusion.cu is a **from-scratch implementation of a Latent Diffusion Model** (the architecture behind Stable Diffusion) where the three most compute-intensive neural-network primitives — **Conv2d, GroupNorm, and Multi-Head Attention** — are replaced with hand-written CUDA kernels that are compiled as PyTorch C++ extensions via pybind11. The system follows the standard Stable Diffusion pipeline: a **CLIP text encoder** converts a prompt into context embeddings, a **U-Net** iteratively denoises a latent tensor conditioned on that context and a sinusoidal time embedding, and a **VAE decoder** upsamples the 64×64 latent back to a 512×512 pixel image. The sampling loop uses the **DDPM** (Denoising Diffusion Probabilistic Models) scheduler with classifier-free guidance. The key engineering contribution is at the **GPU kernel level**: vectorised 128-bit loads (`float4`), tiled shared-memory convolution, cooperative-groups warp reductions for GroupNorm (achieving a **1.85× speedup over PyTorch**), and an online-softmax Flash Attention kernel that keeps the full S×S attention matrix out of global memory.

---

## 2. The Elevator Pitch — *Hook → Action → Result*

> **Hook:** "Diffusion models like Stable Diffusion generate incredible images, but the core operations—convolution, normalization, and attention—spend most of their time moving data through GPU memory rather than doing actual math. I wanted to understand exactly where those bottlenecks are and see if I could beat PyTorch's built-in kernels by writing my own."
>
> **Action:** "I built the entire Latent Diffusion pipeline from scratch in Python and PyTorch—CLIP encoder, U-Net with residual and attention blocks, VAE encoder/decoder, DDPM sampler—and then replaced the three hottest operators with custom CUDA kernels. For GroupNorm, I used float4 vectorised loads and a two-pass cooperative-groups warp reduction. For Conv2d, I tiled the input into shared memory with a halo region to handle the 3×3 filter without extra global reads. For attention, I implemented Flash Attention with an online softmax, processing query and key-value tiles out of shared memory so the full S×S score matrix never materialises in HBM."
>
> **Result:** "The GroupNorm kernel alone achieved a 1.85× speedup over PyTorch's native implementation, and the tiled Conv2d and Flash Attention kernels both demonstrated measurable improvements in memory throughput. More importantly, the project gave me deep, hands-on understanding of GPU memory hierarchies, warp-level programming, and how to profile and optimise the operators that dominate training and inference in modern deep learning."

---

## 3. Amazon-Style Interview Q&A

---

### Q1 — *Dive Deep* · Technical Architecture

**"Walk me through the end-to-end architecture of your diffusion pipeline. Where do your custom CUDA kernels plug in?"**

**Model Answer:**

The pipeline has four major stages, and my CUDA kernels touch three of the four:

1. **Text Encoding (CLIP):** A 12-layer Transformer encoder converts a tokenised prompt into a `(B, 77, 768)` context tensor. Each CLIP layer's self-attention call goes through my Flash Attention kernel.
2. **Iterative Denoising (U-Net):** The U-Net has an encoder path (downsampling from 320→640→1280 channels) , a bottleneck, and a symmetric decoder path with skip connections. Every `UNET_ResidualBlock` calls my **GroupNorm** and **Conv2d** kernels twice each, and every `UNET_AttentionBlock` calls my **Flash Attention** kernel for both self-attention and cross-attention (with the CLIP context). These blocks are executed at *every* timestep (50 steps at inference), so any kernel-level speedup compounds multiplicatively.
3. **VAE Decode:** The decoder mirrors the encoder: Conv2d → ResidualBlock → Upsample, again using my custom GroupNorm and Conv2d.
4. **DDPM Sampler:** Pure Python/PyTorch; schedules the noise variance and computes the denoising step. No custom CUDA here because it's already memory-light scalar math.

The glue between Python and CUDA is **pybind11**: each `.cu` file is compiled as a shared-object module (`flash_atten_kernel.so`, `groupNorm_kernel.so`, `Conv2d_cuda_kernel.so`) and imported like any Python module. A thin Python wrapper class (e.g., `GroupNorm(nn.Module)`) calls the kernel via `data_ptr()` to pass raw GPU pointers.

---

### Q2 — *Dive Deep* · CUDA Optimisation

**"Your GroupNorm kernel achieved a 1.85× speedup over PyTorch. Walk me through the specific optimisations that made that possible."**

**Model Answer:**

GroupNorm is a **memory-bandwidth-bound** operation—it's essentially a statistics computation (mean, variance) plus a normalisation pass. My approach to beating PyTorch focused on three things:

1. **Vectorised Memory Access (`float4`):** Each thread loads and stores 4 floats (128 bits) at a time, which saturates the memory bus far better than scalar loads. This alone can nearly double effective bandwidth on aligned data.
2. **Two-Phase Warp Reduction with Cooperative Groups:** Rather than using `__syncthreads()` plus a shared-memory tree reduction (which serialises across warps), I use `cg::reduce(warp, ..., cg::plus<float>{})` that maps to hardware-accelerated shuffle instructions (`__shfl_xor`). Phase 1 reduces per-warp sums into shared memory (one entry per warp). Phase 2 has warp-0 reduce those partial sums into the final mean and `rsqrtf(var + ε)`. Only two `__syncthreads()` barriers are needed for the entire kernel.
3. **Single-Kernel Fusion:** PyTorch's GroupNorm typically launches separate kernels for computing statistics and applying the normalisation. I fuse both into a single launch, avoiding an extra round-trip to global memory. The normalisation pass simply reloads the input using the same `float4` pattern, applies `(x − mean) × inv_std`, and writes directly to `out`.

The key insight is that GroupNorm's group size on the U-Net channels (32 channels per group) creates blocks of data that fit comfortably into a CUDA block's register and shared-memory budget, making it an ideal candidate for hand-optimisation.

---

### Q3 — *Dive Deep* · Flash Attention Design

**"Why did you implement Flash Attention from scratch rather than using an existing library like Tri Dao's? What were the key design decisions?"**

**Model Answer:**

The primary motivation was **educational depth**—I wanted to fully understand the online-softmax algorithm and the tiled Q/K/V access pattern, not just call an API. But there were also practical design reasons:

1. **Layout Control:** My model stores tensors as `(B, S, H, D)` rather than `(B, H, S, D)`. Using Flash Attention 2's library would have required permutation copies or adapting to its expected layout. By writing my own kernel, I directly index into my layout, avoiding transpose overhead.
2. **Tile Size Tuning:** I chose `TILE_SIZE_Q = 4` (one query row per warp, 4 warps per block) and `TILE_SIZE_KV = 16`. This was tuned for the moderate sequence lengths in Stable Diffusion (e.g., 64×64 = 4096 pixels at the lowest resolution). A production Flash Attention library optimises for much longer sequences (8K+) with larger tiles, but that would over-allocate shared memory on my P100 target (compute capability 6.0).
3. **Online Softmax:** The kernel maintains running `max_val` and `l` (log-sum-exp denominator) per query row across KV tiles, rescaling the accumulator with `exp(prev_max − new_max)` after each tile. This is the core of Flash Attention: the full `S × S` score matrix never exists in HBM; only `TILE_KV` scores live in registers at any point.
4. **Warp-Level Dot Product:** The Q·K dot product is computed by distributing the D-dimension across the 32 lanes of a warp and then doing a `cg::reduce(warp, dot, plus)`, which compiles down to 5 rounds of `__shfl_xor`. No shared memory is needed for the reduction itself.

The trade-off I accepted: my kernel is **forward-only** (no backward pass), which is acceptable for inference but means I can't use it during training. A production kernel would need to store the LSE (log-sum-exp) values for the backward recomputation.

---

### Q4 — *Ownership* · Building From Scratch

**"Why did you choose to build the entire diffusion model from scratch instead of using Hugging Face's `diffusers` library?"**

**Model Answer:**

I had two goals that Hugging Face couldn't serve simultaneously:

1. **Drop-in Kernel Replacement:** I needed full control over how PyTorch operators are called at the Python level so I could swap `nn.Conv2d` → my `Conv2d_K3` wrapper, `nn.GroupNorm` → my `GroupNorm` wrapper, and `F.scaled_dot_product_attention` → my `flash_atten_kernel.flash_atten()`. In Hugging Face's `diffusers`, these calls are buried deep in compiled library code and are not easily interposable.
2. **End-to-End Understanding:** I wanted to reason about tensor shapes, memory layout, and data flow from CLIP tokenisation through DDPM sampling. Building the VAE encoder/decoder, U-Net, CLIP encoder, and DDPM sampler from scratch forced me to understand every dimension annotation—e.g., why the VAE bottleneck is `(B, 4, H/8, W/8)` and how the 0.18215 scaling constant comes from the pretrained checkpoint.

The ownership trade-off: I accept the risk of bugs (and I did discover alignment bugs with `float4` loads on non-16-byte-aligned tensors), but I now understand the system at a level where I can debug any layer's numerical output by hand.

---

### Q5 — *Deliver Results* · Measuring Impact

**"How did you benchmark and validate your CUDA kernels? How do you know the 1.85× speedup is real and not a measurement artefact?"**

**Model Answer:**

I followed a rigorous benchmarking methodology:

1. **Correctness First:** Before any timing, I ran each kernel against PyTorch's reference output with `torch.allclose(atol=1e-4, rtol=1e-3)` across a range of input shapes to confirm numerical equivalence. GroupNorm's `rsqrtf` and Flash Attention's `expf` can accumulate floating-point divergence, so I used task-appropriate tolerances.
2. **Warm-up + Cuda Events:** I used `cudaEvent` timing (not `time.time()`) to measure kernel-only latency. Each benchmark runs 10 warm-up iterations to fill caches and prime the GPU clock, followed by 100 timed iterations. I report the **median** wall time, not the mean, to exclude outlier GC pauses.
3. **Apples-to-Apples Comparison:** The baseline is `torch.nn.GroupNorm` on the **same GPU, same tensor shape, same dtype (float32), same batch size**. I ran both configs on an NVIDIA P100 with compute capability 6.0, which is the architecture my kernel is compiled for.
4. **Throughput Metric:** Beyond raw latency, I computed **effective bandwidth** (bytes read + written / time) to confirm my kernel is closer to the P100's theoretical memory bandwidth (~732 GB/s), which validates that the vectorised loads are achieving memory-bus saturation.

The 1.85× speedup is real and comes primarily from the `float4` vectorisation and the kernel fusion (single-pass stats + normalise), which PyTorch's implementation didn't do at the time of my benchmarking.

---

### Q6 — *Conflict / Trade-off* · Technical Decision

**"Your Conv2d kernel uses `atomicAdd` to accumulate partial sums across input-channel chunks of 16. That's a known performance bottleneck. Why did you keep it?"**

**Model Answer:**

This was an explicit trade-off between **occupancy** and **atomic contention**:

- **The Problem:** A 3×3 Conv2d with C_in=1280 channels (the U-Net bottleneck) would need `1280 × 9 = 11,520` floats of weight in shared memory per block if each block handles all input channels. That's ~45 KB—already exceeding the 48 KB shared-memory limit per SM on P100.
- **My Solution:** I chunk the input channels into groups of 16 (`cin_chunk = 16`). Each thread block handles one `(C_out, cin_chunk=16)` slice and accumulates its partial `c_out_temp` into global memory via `atomicAdd`. With `C_in / 16` blocks writing to the same output pixel, there are at most 80 atomic collisions per output element (for C_in=1280).
- **Why not Reduce?:** An alternative is to use a second "reduction" kernel to sum the partial output planes. This avoids atomics but adds a full kernel launch with its own global memory read/write of the output tensor. On P100, the atomic path was faster for my channel counts because the L2 cache absorbs most atomic traffic when the working set fits in the ~4 MB L2.
- **Future Path:** On Ampere+ GPUs, I would replace `atomicAdd` with `__reduce_add_sync` or use the asynchronous copy (`cp.async`) + shared-memory reduction pattern, which eliminates global-memory contention entirely.

I kept the atomic path because **it was measurably faster for my target hardware and shapes**, and I documented the trade-off in the code comments so a future engineer can swap it out transparently.

---

### Q7 — *Leadership Principle Mix* · Bias for Action / Learn and Be Curious

**"If you had two more weeks to work on this project, what would you prioritise?"**

**Model Answer:**

Three concrete things, in priority order:

1. **FP16 / Mixed Precision Support (3 days):** My kernels currently operate in `float32`. Adding `__half` support with `__hfma2` (half-precision fused multiply-add on pairs of fp16 values) would double the arithmetic throughput and halve memory traffic. This is the single highest-impact optimisation for inference latency.
2. **Backward Passes for Training (5 days):** My Flash Attention kernel is forward-only. Implementing the backward pass (which needs to recompute the attention matrix from the stored LSE values) would make the project usable for fine-tuning, not just inference. This aligns with Flash Attention 2's approach—save `O(S)` metadata instead of `O(S²)` scores.
3. **Nsight Compute Profiling Pass (2 days):** I would run every kernel through NVIDIA Nsight Compute to get exact metrics on **achieved occupancy**, **warp stall reasons**, and **memory-bank conflicts**, then tune TILE_SIZE, BLOCK_SIZE, and register pressure accordingly. Right now my tile sizes are based on reasonable heuristics; profiling would convert them into data-driven choices.

These improvements would transform the project from an educational demonstrator into something approaching a production-grade inference engine.

---

## Quick Reference — Amazon Leadership Principle Mapping

| # | Question Theme | Primary LP | Secondary LP |
|---|---|---|---|
| Q1 | End-to-end architecture | **Dive Deep** | Learn and Be Curious |
| Q2 | GroupNorm 1.85× speedup | **Dive Deep** | Deliver Results |
| Q3 | Flash Attention from scratch | **Dive Deep** | Ownership |
| Q4 | Build vs. use library | **Ownership** | Learn and Be Curious |
| Q5 | Benchmarking methodology | **Deliver Results** | Insist on Highest Standards |
| Q6 | atomicAdd trade-off | **Have Backbone; Disagree and Commit** | Dive Deep |
| Q7 | Future priorities | **Bias for Action** | Learn and Be Curious |
