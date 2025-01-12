import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_kernel_inner(
    O_block,                    # (BLOCK_SIZE_Q, HEAD_DIM)
    l_i, m_i,                   # (BLOCK_SIZE_Q,)
    Q_block,                    # (BLOCK_SIZE_Q, HEAD_DIM)
    K_block_ptr,                # pointer to (HEAD_DIM, BLOCK_SIZE_KV)
    V_block_ptr,                # pointer to (BLOCK_SIZE_KV, HEAD_DIM)
    block_index_q,              # int
    softmax_scale,              # float
    offset_q: tl.constexpr,     # (BLOCK_SIZE_Q,)
    offset_kv: tl.constexpr,    # (BLOCK_SIZE_KV,)
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # causal 
    if STAGE == 1:
        # From 0 to the left of the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        # let the compiler know that lo is a multiple of BLOCK_SIZE_Q, so the compiler can do optimizations
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # non-causal attention
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo)) # K_block is (HEAD_DIM, BLOCK_SIZE_KV)
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0)) # V_block is (BLOCK_SIZE_KV, HEAD_DIM)

    # loop over K, V and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # let the compiler know that start_kv is a multiple of BLOCK_SIZE_KV, so the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # load K, V
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)
        
        if STAGE == 2:
            mask = offset_q[:, None] >= (start_kv + offset_kv[None, :])
            QK_block = QK_block * softmax_scale
            QK_block = tl.where(mask, QK_block, -float("inf"))
            
            m_ij = tl.maximum(m_i, tl.max(QK_block, axis=1))
            QK_block = QK_block - m_ij[:, None]
        else:
            QK_block = QK_block * softmax_scale
            m_ij = tl.maximum(m_i, tl.max(QK_block, axis=1))
            QK_block = QK_block - m_ij[:, None]

        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)
        # Compute the sum of the exponential values for each row
        l_ij = tl.sum(P_block, axis=1)
        
        # correction factor for previous iteration alpha
        alpha = tl.math.exp(m_i - m_ij)
        # apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)

        # Compute O wtih corrections applied - O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, acc=O_block)

        m_i = m_ij

        # Advance to next K, V blocks
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
    
    return O_block, l_i, m_i


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V,                            # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,                      # float
    L,                                  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,                                  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    stride_Q_batch, stride_Q_head, stride_Q_seq, stride_Q_dim, # int
    stride_K_batch, stride_K_head, stride_K_seq, stride_K_dim, # int
    stride_V_batch, stride_V_head, stride_V_seq, stride_V_dim, # int
    stride_O_batch, stride_O_head, stride_O_seq, stride_O_dim, # int
    # batch size is not kept constexpr as it is user defined - kernel should not be recompiled for every new batch size
    BATCH_SIZE,
    # these variables are kept as constexpr as these are usually fixed for a model so can be baked into the compilation
    NUM_HEADS: tl.constexpr, SEQ_LEN: tl.constexpr, HEAD_DIM: tl.constexpr, 
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):

    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM, "BLOCK_SIZE_KV must be less than or equal to HEAD_DIM")

    # Which block of Q are we processing?
    block_index_q = tl.program_id(0)
    # Which batch and head are we processing?
    index_batch_head = tl.program_id(1)

    # Get exact index for batch and head
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    # Move past the "covered" batch and head
    # Below expressions are equivalent
    # Since, index_batch_head = index_batch * NUM_HEADS + index_head
    # => index_batch_head * SEQ_LEN * HEAD_DIM 
    # = (index_batch * NUM_HEADS * SEQ_LEN * HEAD_DIM)  + (index_head * SEQ_LEN * HEAD_DIM)
    # = (index_batch * stride_batch) + (index_head * stride_head)
    qkv_offset = index_batch_head.to(tl.int64) * SEQ_LEN * HEAD_DIM
    # qkv_offset = (index_batch.to(tl.int64) * stride_Q_batch + index_head.to(tl.int64) * stride_Q_head)

    Q_block_ptr = tl.make_block_ptr(
        base = Q + qkv_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_Q_seq, stride_Q_dim),
        offsets = (block_index_q * BLOCK_SIZE_Q, 0),
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        base = K + qkv_offset,
        shape = (HEAD_DIM, SEQ_LEN),
        # invert strides since we are loading K^T
        strides = (stride_K_dim, stride_K_seq),
        offsets = (0, 0),
        block_shape = (HEAD_DIM, BLOCK_SIZE_KV),
        order = (0, 1)
    )

    V_block_ptr = tl.make_block_ptr(
        base = V + qkv_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_V_seq, stride_V_dim),
        offsets = (0, 0),
        block_shape = (BLOCK_SIZE_KV, HEAD_DIM),
        order = (1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        base = O + qkv_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_O_seq, stride_O_dim),
        offsets = (block_index_q * BLOCK_SIZE_Q, 0),
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0)
    )

    # offset for tokens in Q and K,V sequences to process
    offset_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offset_kv = tl.arange(0, BLOCK_SIZE_KV)

    # running maximum
    m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32) - float("inf")

    # running sum
    l_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32) + 1.0

    # the accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)

    # load the block of Q (outer loop) => this will stay in SRAM
    Q_block = tl.load(Q_block_ptr)

    # stage = 3 if causal else 1
    if STAGE == 1 or STAGE == 3:
        # this step runs for non-causal attention (for stage = 1) or for 
        # the blocks to the left of the diagonal in causal attention
        # since diagonal is also a block => some parts of the diagonal
        # be a part of the attention and some won't
        # (covered in case of stage = 3 in next condition)

        O_block, l_i, m_i = _attn_fwd_kernel_inner(
            O_block,        # (BLOCK_SIZE_Q, HEAD_DIM)
            l_i, m_i,       # (BLOCK_SIZE_Q,)
            Q_block,        # (BLOCK_SIZE_Q, HEAD_DIM)
            K_block_ptr,    # pointer to (HEAD_DIM, BLOCK_SIZE_KV)
            V_block_ptr,    # pointer to (BLOCK_SIZE_KV, HEAD_DIM)
            block_index_q,  # int
            softmax_scale,  # float
            offset_q,       # (BLOCK_SIZE_Q,)
            offset_kv,      # (BLOCK_SIZE_KV,)
            BLOCK_SIZE_Q, BLOCK_SIZE_KV,
            4 - STAGE,
            SEQ_LEN
        )

    if STAGE == 3:
        # this step runs for the block in which there is transition between non-masked and masked keys
        # this is the only block that has to be computed in this step
        # the rest of the blocks are computed in previous condition
        O_block, l_i, m_i = _attn_fwd_kernel_inner(
            O_block,        # (BLOCK_SIZE_Q, HEAD_DIM)
            l_i, m_i,       # (BLOCK_SIZE_Q,)
            Q_block,        # (BLOCK_SIZE_Q, HEAD_DIM)
            K_block_ptr,    # pointer to (HEAD_DIM, BLOCK_SIZE_KV)
            V_block_ptr,    # pointer to (BLOCK_SIZE_KV, HEAD_DIM)
            block_index_q,  # int
            softmax_scale,  # float
            offset_q,       # (BLOCK_SIZE_Q,)
            offset_kv,      # (BLOCK_SIZE_KV,)
            BLOCK_SIZE_Q, BLOCK_SIZE_KV,
            2,
            SEQ_LEN
        )

    # finalizing outputs
    # Divide by softmax normalizing factor
    O_block = O_block / l_i[:, None]
    # Store log-sum-exp
    # During backward pass we can simply do 
    # softmax(x) = exp(x - L)
    # => exp(x - m - log(l))
    # => exp(x - m) / exp(log(l))
    # => exp(x - m) / l
    L_i = m_i + tl.math.log(l_i)
    L_ptrs = L + index_batch_head * SEQ_LEN + offset_q

    tl.store(L_ptrs, L_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))