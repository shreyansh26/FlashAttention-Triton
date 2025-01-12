# FlashAttention in Triton

An implementation of [FlashAttention](https://arxiv.org/abs/2307.08691) using Triton. This is based on the [official Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) and [Umar Jamil's awesome talk](https://www.youtube.com/watch?v=zy8ChVd_oTM).
I have added additional comments and proofs to aid my understanding of the code.

The code in the Triton tutorials is more performant. This implementation is only for educational purposes.

## Requirements
This has been tested on -
```
torch==2.4.0
triton==3.0.0
```

## Files
* [fwd.py](fwd.py) - Implementation of the Triton kernel for the forward pass of FlashAttention. It supports both causal and non-causal implementations.
* [bwd.py](bwd.py) - Implementation of the Triton kernel for the backward pass of FlashAttention. It supports both causal and non-causal implementations.
* [flash_attention.py](flash_attention.py) - The host-side torch function for the kernel which takes in Q, K, V, the softmax scale and a boolean flag for causal/non-causal.
* [test_correctness.py](test_correctness.py) - Includes a simple test of equivalence of the forward pass (both causal and non-causal) with normal attention along with benchmarking.

## To run
    
```python test_correctness.py```