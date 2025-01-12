import torch
import triton
from fwd import _attn_fwd_kernel
from bwd import _attn_bwd_preprocess_kernel, _attn_bwd_dk_dv_kernel, _attn_bwd_dq_kernel

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
        ctx.HEAD_DIM = HEAD_DIM_K

        return O_bhsd


    @staticmethod
    def backward(ctx, dO_bhsd):
        Q_bhsd, K_bhsd, V_bhsd, O_bhsd, L_bhs = ctx.saved_tensors

        assert dO_bhsd.is_contiguous()
        assert dO_bhsd.stride() == Q_bhsd.stride() == K_bhsd.stride() == V_bhsd.stride() == O_bhsd.stride()

        dQ_bhsd = torch.empty_like(Q_bhsd)
        dK_bhsd = torch.empty_like(K_bhsd)
        dV_bhsd = torch.empty_like(V_bhsd)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, _ = Q_bhsd.shape
        HEAD_DIM = ctx.HEAD_DIM
        
        preprocess_grid = lambda args: (triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS, 1)

        D_bhs = torch.empty_like(L_bhs)

        # Compute all Di
        _attn_bwd_preprocess_kernel[preprocess_grid](
            O=O_bhsd,
            dO=dO_bhsd,
            D=D_bhs,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
        )

        grid = lambda args: (SEQ_LEN // args["BLOCK_SIZE_KV"], BATCH_SIZE * NUM_HEADS, 1)

        stage = 3 if ctx.causal else 1

        # Fix KV and iterate through all the Q blocks
        _attn_bwd_dk_dv_kernel[grid](
            Q=Q_bhsd,
            K=K_bhsd,
            V=V_bhsd,
            softmax_scale=ctx.softmax_scale,
            dO=dO_bhsd,
            dQ=dQ_bhsd,
            dK=dK_bhsd,
            dV=dV_bhsd,
            L=L_bhs,
            D=D_bhs,
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
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
        )

        grid = lambda args: (SEQ_LEN // args["BLOCK_SIZE_Q"], BATCH_SIZE * NUM_HEADS, 1)

        # Fix Q and iterate through all the KV block
        _attn_bwd_dq_kernel[grid](
            Q=Q_bhsd,
            K=K_bhsd,
            V=V_bhsd,
            softmax_scale=ctx.softmax_scale,
            dO=dO_bhsd,
            dQ=dQ_bhsd,
            dK=dK_bhsd,
            dV=dV_bhsd,
            L=L_bhs,
            D=D_bhs,
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
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
        )

        return dQ_bhsd, dK_bhsd, dV_bhsd, None, None