import torch
from torch import nn
import flash_atten_kernel

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask = False):
        """
        params needed to call cuda kernel flash attention:
        uintptr_t Q_ptr,
        uintptr_t K_ptr, 
        uintptr_t V_ptr,
        uintptr_t Out_ptr,
        uintptr_t LSE_ptr,
        int B, int S, int H, int D,
        bool is_causal
        """
        # x: (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) ||  3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H)
        q = q.view(interim_shape).contiguous()
        k = k.view(interim_shape).contiguous()
        v = v.view(interim_shape).contiguous()

        lse = torch.zeros((batch_size, self.n_heads, sequence_length), dtype=torch.float32, device=x.device)
        custom_out = torch.zeros_like(q)

        flash_atten_kernel.flash_atten(
            q.data_ptr(), k.data_ptr(), v.data_ptr(), 
            custom_out.data_ptr(), lse.data_ptr(),  
            batch_size, sequence_length, self.n_heads, self.d_head,
            causal_mask
        )

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = custom_out.reshape(input_shape) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len, Dim)
        return output
    


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)  ## no bias  ||   it takes x 
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)  # takes y
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)  # takes y
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads    #  per head dim   


    def forward(self, x, y) :
        # x: (Batch_Size, Seq_Len_Q, Dim), y: (Batch_Size, Seq_Len_KV, Dim)
        input_shape = x.shape
        batch_size, seq_len_q, d_embed = input_shape 
        batch_size_kv, seq_len_kv, d_cross = y.shape

        # B, S, H, D 
        # Use separate shapes for Q and KV
        q_shape = (batch_size, seq_len_q, self.n_heads, self.d_head)
        kv_shape = (batch_size_kv, seq_len_kv, self.n_heads, self.d_head)

        q = self.q_proj(x).view(q_shape).contiguous()
        k = self.k_proj(y).view(kv_shape).contiguous()
        v = self.v_proj(y).view(kv_shape).contiguous()

        lse = torch.zeros((batch_size, self.n_heads, sequence_length), dtype=torch.float32, device=x.device)
        custom_out = torch.zeros_like(q)

        flash_atten_kernel.flash_atten(
            q.data_ptr(), k.data_ptr(), v.data_ptr(),
            custom_out.data_ptr(), lse.data_ptr(),
            batch_size, seq_len_q, self.n_heads, self.d_head,
            False    # CrossAttention should typically not be causal
        )
        
        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = custom_out.reshape(input_shape)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len, Dim)
        return output
    



