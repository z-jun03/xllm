# Release xllm 0.5.0

## **Major Features and Improvements**

- Support NPU operators of the RC1 version.
- Support disaggregated prefill and decoding.
- Support large-scale EP parallelism.
- Support Hash-based PrefixCache matching.
- Support Multi-Token Prediction for DeepSeek.
- Replace Prometheus with bvar to enhance metrics collection performance.
- Optimize greedy sampling performance.
- Support Qwen3 model on NPU.

# Release xllm 0.4.0

## **Major Features and Improvements**

- Support NPU operators of the T9 version to enhance the performance of MLA and MOE operators.
- Support asynchronous scheduling, allowing the scheduling and computational pipeline to execute in parallel.
- Support multiple nodes on NPU.

# Release xllm 0.3.1

## **Major Features and Improvements**

- Support the metrics interface for BRPC services.
- Fix the issue of inference failure caused by incorrect model input construction.

# Release xllm 0.3.0

## **Major Features and Improvements**

- Support DeepSeek-V2, DeepSeek-V3, DeepSeek-R1 models on GPU and NPU.
- Support DeepSeek distilled models, such as DeepSeek-R1-Distill-Qwen on GPU and NPU.
- Support QwQ-32B, Qwen2.5 models on GPU and NPU.
- Support Chatrhino model on GPU and NPU.
- Add MoE triton ops in GPU.
- Add FlashMLA and flashinfer for MLA ops.
- Add MoE and MLA's NPU ops.
- Support EP, DP, TP model parallel. (EP and DP only support on NPU T6 package.)
- Support multiple process and multiple nodes.
- Support for BRPC-based request http services.