# chunked调度器

## 功能介绍
xLLM支持chunked prefill调度策略。Chunked prefill是一种优化大语言模型推理的技术，将长prompt分割成多个较小的chunk进行分批处理，而不是一次性处理整个prompt。
这种方法可以有效降低显存峰值使用量，提高Device利用率，并且能够更好地与decode阶段的请求进行调度和混合处理。

## 使用方式
上述策略已在xLLM实现，并向外暴露gflag参数，控制功能的开关。

- 开启chunked prefill，并设置chunked_size。
```
--enable_chunked_prefill=true
--max_tokens_per_chunk_for_prefill=256
```

## 性能效果
开启chunked_prefill之后，在Qwen3-8B模型上，限制TPOT 50ms，TTFT时延 **下降46%**。
