# Environment

## Environment

```
source .env/activate
python -c "import triton.profiler as proton"
```

## Commands

Single GPU

```
python generate_proton.py --compile
```

Multi-GPU

```
ENABLE_INTRA_NODE_COMM=1 \
torchrun --standalone --nproc_per_node=4 generate_proton.py --compile \
    --checkpoint_path "checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"
```

```
ENABLE_INTRA_NODE_COMM=1 \
torchrun --standalone --nproc_per_node=4 generate_proton.py --compile \
    --checkpoint_path "checkpoints/meta-llama/Llama-2-7b-chat-hf/model_int8.pth"
```

## Results

### llama-2-7b

#### Single A100
```bash
Time for inference 5: 1.88 sec total, 106.35 tokens/sec
Bandwidth achieved: 1405.39 GB/s
```

```
3.082 decode_one_token
   ├─ 1.322 _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEENS_6detail5ArrayIPcLi1EEEEEviT0_T1_
   ├─ 0.465 triton_red_fused_div_mm_21
   ├─ 0.266 triton_red_fused_mm_1
   ├─ 0.191 triton_red_fused_mm_10
   ├─ 0.454 triton_red_fused_mm_15
   └─ 0.370 triton_red_fused_mm_17
```

#### 4 A100 

```bash
Time for inference 5: 0.83 sec total, 241.81 tokens/sec
Bandwidth achieved: 846.51 GB/s
```

`proton-viewer -m time/s,tbytes/s llama-2-7b-generate-shadow-rank0.hatchet -t 0.01`
```
4.746 ROOT
├─ 0.514 _ZN4c10d15intra_node_comm22oneShotAllReduceKernelILj4ELb1EEEvPN3c108BFloat16EmmPPNS0_8P2pStateEPS4_mb
└─ 4.173 decode_one_token
   ├─ 0.698 _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEENS_6detail5ArrayIPcLi1EEEEEviT0_T1_
   ├─ 2.866 _ZN4c10d15intra_node_comm22oneShotAllReduceKernelILj4ELb1EEEvPN3c108BFloat16EmmPPNS0_8P2pStateEPS4_mb
   ├─ 0.462 triton_red_fused_div_mm_20
   ├─ 0.076 triton_red_fused_mm_10
   └─ 0.059 triton_red_fused_mm_16
```


`proton-viewer -m time/s,tbytes/s llama-2-7b-generate-shadow-rank1.hatchet -t 0.01`
```
2.925 ROOT
├─ 0.442 _ZN4c10d15intra_node_comm22oneShotAllReduceKernelILj4ELb1EEEvPN3c108BFloat16EmmPPNS0_8P2pStateEPS4_mb
└─ 2.425 decode_one_token
   ├─ 1.181 _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEENS_6detail5ArrayIPcLi1EEEEEviT0_T1_
   ├─ 0.457 _ZN4c10d15intra_node_comm22oneShotAllReduceKernelILj4ELb1EEEvPN3c108BFloat16EmmPPNS0_8P2pStateEPS4_mb
   ├─ 0.461 triton_red_fused_div_mm_20
   ├─ 0.098 triton_red_fused_mm_10
   ├─ 0.128 triton_red_fused_mm_14
   └─ 0.089 triton_red_fused_mm_16
```

`proton-viewer -m time/s,tbytes/s llama-2-7b-generate-shadow-rank2.hatchet -t 0.01`
```
3.091 ROOT
├─ 0.183 _ZN4c10d15intra_node_comm22oneShotAllReduceKernelILj4ELb1EEEvPN3c108BFloat16EmmPPNS0_8P2pStateEPS4_mb
└─ 2.849 decode_one_token
   ├─ 1.014 _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEENS_6detail5ArrayIPcLi1EEEEEviT0_T1_
   ├─ 1.095 _ZN4c10d15intra_node_comm22oneShotAllReduceKernelILj4ELb1EEEvPN3c108BFloat16EmmPPNS0_8P2pStateEPS4_mb
   ├─ 0.460 triton_red_fused_div_mm_20
   ├─ 0.076 triton_red_fused_mm_10
   ├─ 0.139 triton_red_fused_mm_14
   └─ 0.053 triton_red_fused_mm_16
```

`proton-viewer -m time/s,tbytes/s llama-2-7b-generate-shadow-rank3.hatchet -t 0.01`
```
hatchet -t 0.01
4.344 ROOT
├─ 0.264 _ZN4c10d15intra_node_comm22oneShotAllReduceKernelILj4ELb1EEEvPN3c108BFloat16EmmPPNS0_8P2pStateEPS4_mb
└─ 4.022 decode_one_token
   ├─ 0.660 _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEENS_6detail5ArrayIPcLi1EEEEEviT0_T1_
   ├─ 2.767 _ZN4c10d15intra_node_comm22oneShotAllReduceKernelILj4ELb1EEEvPN3c108BFloat16EmmPPNS0_8P2pStateEPS4_mb
   ├─ 0.457 triton_red_fused_div_mm_20
   ├─ 0.077 triton_red_fused_mm_10
   └─ 0.051 triton_red_fused_mm_16
```

PyTorch profiler rank0
```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  --------------------------------------------------------------------------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls                                                                      Input Shapes   Total FLOPs  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  --------------------------------------------------------------------------------  ------------  
                                  Torch-Compiled Region         2.11%      16.906ms        67.09%     537.640ms       2.702ms     736.751ms        98.15%     737.950ms       3.708ms           199                                                                                []            --  
                                 triton_red_fused_mm_10         0.00%       0.000us         0.00%       0.000us       0.000us     112.062ms        14.93%     112.062ms      17.598us          6368                                                                                []            --  
                                 triton_red_fused_mm_14         0.00%       0.000us         0.00%       0.000us       0.000us      92.314ms        12.30%      92.314ms      28.993us          3184                                                                                []            --  
                                  triton_red_fused_mm_9         0.00%       0.000us         0.00%       0.000us       0.000us      87.107ms        11.60%      87.107ms      27.358us          3184                                                                                []            --  
void c10d::intra_node_comm::oneShotAllReduceKernel<4...         0.00%       0.000us         0.00%       0.000us       0.000us      69.323ms         9.23%      69.323ms       5.416us         12800                                                                                []            --  
                                 triton_per_fused_bmm_7         0.00%       0.000us         0.00%       0.000us       0.000us      60.117ms         8.01%      60.117ms       9.440us          6368                                                                                []            --  
                                 triton_red_fused_mm_16         0.00%       0.000us         0.00%       0.000us       0.000us      58.642ms         7.81%      58.642ms      19.645us          2985                                                                                []            --  
                                 triton_red_fused_mm_12         0.00%       0.000us         0.00%       0.000us       0.000us      53.995ms         7.19%      53.995ms      16.958us          3184                                                                                []            --  
                                 triton_red_fused_bmm_5         0.00%       0.000us         0.00%       0.000us       0.000us      52.308ms         6.97%      52.308ms       8.214us          6368                                                                                []            --  
                             triton_red_fused_div_mm_20         0.00%       0.000us         0.00%       0.000us       0.000us      30.749ms         4.10%      30.749ms     154.515us           199                                                                                []            --  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  --------------------------------------------------------------------------------  ------------  
```