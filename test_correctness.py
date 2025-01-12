import torch
from flash_attention import FlashAttentionTriton

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    # dO = torch.randn_like(Q)

    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    # ref_O.backward(dO)
    # ref_dV, V.grad = V.grad.clone(), None
    # ref_dK, K.grad = K.grad.clone(), None
    # ref_dQ, Q.grad = Q.grad.clone(), None

    # triton implementation
    tri_out = FlashAttentionTriton.apply(Q, K, V, softmax_scale, causal).half()
    # tri_out.backward(dO)
    # tri_dV, V.grad = V.grad.clone(), None
    # tri_dK, K.grad = K.grad.clone(), None
    # tri_dQ, Q.grad = Q.grad.clone(), None

    # compare
    rtol = 0.0
    atol = 1e-2
    
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    # assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    # assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    # assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)


if __name__ == "__main__":
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=True)
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=False)
    print("PASSED")