import torch
import triton
from fwd import _attn_fwd_kernel

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q_bhsd, K_bhsd, V_bhsd, softmax_scale, causal):
        HEAD_DIM_Q = Q_bhsd.shape[-1]
        HEAD_DIM_K = K_bhsd.shape[-1]
        HEAD_DIM_V = V_bhsd.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, _ = Q_bhsd.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O_bhsd = torch.empty_like(Q_bhsd)

        stage = 3 if causal else 1

        grid = lambda args: (triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS, 1)

        # L is the log-sum-exp for the backward pass
        L_bhs = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN, device=Q_bhsd.device, dtype=torch.float32)

        _attn_fwd_kernel[grid](
            Q=Q_bhsd,
            K=K_bhsd,
            V=V_bhsd,
            softmax_scale=softmax_scale,
            L=L_bhs,
            O=O_bhsd,
            stride_Q_batch=Q_bhsd.stride(0),
            stride_Q_head=Q_bhsd.stride(1),
            stride_Q_seq=Q_bhsd.stride(2),
            stride_Q_dim=Q_bhsd.stride(3),
            stride_K_batch=K_bhsd.stride(0),
            stride_K_head=K_bhsd.stride(1),
            stride_K_seq=K_bhsd.stride(2),
            stride_K_dim=K_bhsd.stride(3),
            stride_V_batch=V_bhsd.stride(0),
            stride_V_head=V_bhsd.stride(1),
            stride_V_seq=V_bhsd.stride(2),
            stride_V_dim=V_bhsd.stride(3),
            stride_O_batch=O_bhsd.stride(0),
            stride_O_head=O_bhsd.stride(1),
            stride_O_seq=O_bhsd.stride(2),
            stride_O_dim=O_bhsd.stride(3),
            BATCH_SIZE=Q_bhsd.shape[0],
            NUM_HEADS=Q_bhsd.shape[1],
            SEQ_LEN=Q_bhsd.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        ctx.save_for_backward(Q_bhsd, K_bhsd, V_bhsd, O_bhsd, L_bhs)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal

        return O_bhsd


    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError
