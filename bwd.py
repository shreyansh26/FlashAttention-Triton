import torch
import triton
import triton.language as tl

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [32, 64, 128]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd_preprocess_kernel(
    O,
    dO,
    D,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    offset_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    index_batch_head = tl.program_id(1)
    offset_dim = tl.arange(0, HEAD_DIM)

    # Load a single block of BLOCK_SIZE_Q rows of O and dO
    O_block = tl.load(O + index_batch_head * SEQ_LEN * HEAD_DIM + offset_q[:, None] * HEAD_DIM + offset_dim[None, :])
    dO_block = tl.load(dO + index_batch_head * SEQ_LEN * HEAD_DIM + offset_q[:, None] * HEAD_DIM + offset_dim[None, :]).to(tl.float32)

    # Compute D block
    # Elementwise multiply dO_block and O_block, sum over the head dimension, and store the result in D_block
    D_block = tl.sum(dO_block * O_block, axis=1)

    # Calculate the pointer to the D block
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offset_q
    # Store D block
    tl.store(D_block_ptrs, D_block)

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [32, 64, 128]
        for BLOCK_SIZE_KV in [32, 64, 128]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd_dk_dv_kernel(
    Q, K, V,
    softmax_scale,
    dO,
    dQ, dK, dV,
    L,
    D,
    stride_Q_batch, stride_Q_head, stride_Q_seq, stride_Q_dim,
    stride_K_batch, stride_K_head, stride_K_seq, stride_K_dim,
    stride_V_batch, stride_V_head, stride_V_seq, stride_V_dim,
    NUM_HEADS: tl.constexpr, SEQ_LEN: tl.constexpr, HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr
):
    index_batch_head = tl.program_id(1)
    offset_batch_head = (index_batch_head * SEQ_LEN * HEAD_DIM).to(tl.int64)

    # Offset to select the right sequence given the batch and head
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Move pointers w.r.t batch and head
    # make_block_ptr is not used to apply masking
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Move pointers w.r.t batch, head, sequence
    L += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offset_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_SIZE_KV
    offset_kv = start_kv + tl.arange(0, BLOCK_SIZE_KV)

    dK_block = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), dtype=tl.float32)
    dV_block = tl.zeros((BLOCK_SIZE_KV, HEAD_DIM), dtype=tl.float32)

    # Load K and V blocks => they will stay in SRAM
    # Shape: (BLOCK_SIZE_KV, HEAD_DIM)
    K_block = tl.load(K + offset_kv[:, None] * HEAD_DIM + offset_dim[None, :])
    V_block = tl.load(V + offset_kv[:, None] * HEAD_DIM + offset_dim[None, :])

    offset_q = tl.arange(0, BLOCK_SIZE_Q)

    # We access the Q as a transposed array, so that's why we treat offset_q as a column vector ans offset_dim as a row vector
    # This is equivalent to doing:
    # q_ptrs = Q + offset_q[:, None] * stride_seq + offset_dim[None, :] * stride_dim
    # qT_ptrs = tl.trans(q_ptrs)
    # We point to the first BLOCK_Q rows of Q for both the qT and dO pointers, inside the for loop we will move forward by BLOCK_Q rows at each iteration.
    Q_T_ptrs = Q + offset_q[None, :] * HEAD_DIM + offset_dim[:, None]
    dO_ptrs = dO + offset_q[:, None] * HEAD_DIM + offset_dim[None, :]

    # Need to reconstruct P
    # Iterate over the sequence dimension of query
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_SIZE_Q

    for step in range(num_steps):
        # Load Q block
        Q_T_block = tl.load(Q_T_ptrs)
        # Load the logsumexp values for the queries in the current block
        offset_q = curr_q + tl.arange(0, BLOCK_SIZE_Q)
        L_block = tl.load(L + offset_q)

        # This gives us (QK^T)^T = (K^T)^T(Q^T) = K(Q^T) = P^T
        QK_T_block = tl.dot(K_block, Q_T_block) * softmax_scale
        # Use logsumexp to get the softmax values
        P_T_block = tl.math.exp(QK_T_block - L_block[None, :])

        if STAGE == 3:
            # Autoregressive masking
            # mask is True for all values that DO NOT NEED TO BE MASKED
            mask_block = (offset_q[None, :] >= offset_kv[:, None])
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        dO_block = tl.load(dO_ptrs)
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        # Load Di which is elementwise product of O and dO
        Di = tl.load(D + offset_q)
        
        # dP = dO x V^T, so dP^T = V x dO^T
        dP_T_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        # dS = P * (dP - Di)
        # dS^T = P^T * (dP^T - D^T)
        dS_T_block = P_T_block * (dP_T_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)

        # dK = dS^T x Q
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(Q_T_block))

        # Increment pointers
        curr_q += BLOCK_SIZE_Q
        Q_T_ptrs += BLOCK_SIZE_Q * HEAD_DIM
        dO_ptrs += BLOCK_SIZE_Q * HEAD_DIM
        
    # Write dV
    dV_block_ptrs = dV + offset_kv[:, None] * HEAD_DIM + offset_dim[None, :]
    tl.store(dV_block_ptrs, dV_block)

    # Write dK
    dK_block_ptrs = dK + offset_kv[:, None] * HEAD_DIM + offset_dim[None, :]
    tl.store(dK_block_ptrs, dK_block)

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [32, 64, 128]
        for BLOCK_SIZE_KV in [32, 64, 128]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd_dq_kernel(
    Q, K, V,                    # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    softmax_scale,              # float
    dO,                         # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    dQ, dK, dV,                 # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    L,                          # (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
    D,                          # (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
    stride_Q_batch, stride_Q_head, stride_Q_seq, stride_Q_dim,
    stride_K_batch, stride_K_head, stride_K_seq, stride_K_dim,
    stride_V_batch, stride_V_head, stride_V_seq, stride_V_dim,
    NUM_HEADS: tl.constexpr, SEQ_LEN: tl.constexpr, HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr
):
    index_batch_head = tl.program_id(1)
    offset_batch_head = (index_batch_head * SEQ_LEN * HEAD_DIM).to(tl.int64)

    # Offset to select the right sequence given the batch and head
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Move pointers w.r.t batch and head
    # make_block_ptr is not used to apply masking
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Move pointers w.r.t batch, head, sequence
    L += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offset_dim = tl.arange(0, HEAD_DIM)

    index_block_q = tl.program_id(0)
    start_q = index_block_q * BLOCK_SIZE_Q
    offset_q = start_q + tl.arange(0, BLOCK_SIZE_Q)

    # Load Q block => this will stay in SRAM
    # Shape: (BLOCK_SIZE_Q, HEAD_DIM)
    Q_block = tl.load(Q + offset_q[:, None] * HEAD_DIM + offset_dim[None, :])
    
    dQ_block = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)
    dO_block = tl.load(dO + offset_q[:, None] * HEAD_DIM + offset_dim[None, :])

    L_block = tl.load(L + offset_q)

    offset_kv = tl.arange(0, BLOCK_SIZE_KV)

    # We access the K, V as a transposed array
    K_T_ptrs = K + offset_kv[None, :] * HEAD_DIM + offset_dim[:, None]
    V_T_ptrs = V + offset_kv[None, :] * HEAD_DIM + offset_dim[:, None]

    Di = tl.load(D + offset_q)

    # Need to reconstruct P
    # Iterate over the sequence dimension of query
    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_SIZE_KV

    for step in range(num_steps):
        # Load Q block
        K_T_block = tl.load(K_T_ptrs)
        V_T_block = tl.load(V_T_ptrs)

        QK_block = tl.dot(Q_block, K_T_block) * softmax_scale
        # Use logsumexp to get the softmax values
        P_block = tl.math.exp(QK_block - L_block[:, None])

        if STAGE == 3:
            offset_kv = curr_kv + tl.arange(0, BLOCK_SIZE_KV)

            # Autoregressive masking
            # mask is True for all values that DO NOT NEED TO BE MASKED
            mask_block = (offset_q[:, None] >= offset_kv[None, :])
            P_block = tl.where(mask_block, P_block, 0.0)

        # Compute dP and dS
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)

        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))

        # Increment pointers
        curr_kv += BLOCK_SIZE_KV
        K_T_ptrs += BLOCK_SIZE_KV * HEAD_DIM
        V_T_ptrs += BLOCK_SIZE_KV * HEAD_DIM
          
    # Write dQ
    dQ_block_ptrs = dQ + offset_q[:, None] * HEAD_DIM + offset_dim[None, :]
    tl.store(dQ_block_ptrs, dQ_block)