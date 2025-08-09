
# Advanced NLP & LLM Systems Roadmap (Production-Focused)

A curated, **production-first** roadmap for advanced NLP/LLM topics—from GPU kernels and KV-cache internals to decoding tricks, quantization, MoE, long-context methods, RAG retrieval, serving stacks, and evaluation. Each topic includes a **brief explainer**, **what you’ll learn**, and **free resources**.

> Who is this for? Senior ML/NLP engineers who want to ship **fast, cheap, and reliable** LLM systems in production.

---

## Table of Contents

1. [Systems Foundations (CUDA, Triton, Memory)](#1-systems-foundations-cuda-triton-memory)
2. [Attention & KV-Cache Internals](#2-attention--kv-cache-internals)
3. [Decoding for Throughput & Quality](#3-decoding-for-throughput--quality)
4. [Quantization (Weights, Activations, KV Cache)](#4-quantization-weights-activations-kv-cache)
5. [Sparsity & Pruning](#5-sparsity--pruning)
6. [Long-Context Methods](#6-long-context-methods)
7. [Streaming & State Space Models](#7-streaming--state-space-models)
8. [PEFT: LoRA & Friends](#8-peft-lora--friends)
9. [Training at Scale (FSDP, ZeRO, Checkpointing)](#9-training-at-scale-fsdp-zero-checkpointing)
10. [Serving Stacks & Compilers](#10-serving-stacks--compilers)
11. [Advanced Retrieval & RAG](#11-advanced-retrieval--rag)
12. [Evaluation & Benchmarking](#12-evaluation--benchmarking)
13. [Production Tips & Checklists](#13-production-tips--checklists)
14. [Suggested Learning Path](#14-suggested-learning-path)

---

## 1) Systems Foundations (CUDA, Triton, Memory)

**Brief**: Master the GPU memory hierarchy, thread/block/wrap scheduling, and kernel fusion. Write custom kernels when PyTorch isn’t enough.

**What you’ll learn**
- How shared/L2/global memory and registers affect throughput
- Kernel fusion patterns (matmul+softmax+scale) and IO-aware designs
- Writing kernels in **Triton** to accelerate bottlenecks

**Free resources**
- NVIDIA **CUDA C++ Programming Guide** (official PDF): https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
- **Triton** language docs & tutorials: https://triton-lang.org/main/index.html

---

## 2) Attention & KV-Cache Internals

**Brief**: Attention is memory-bound; learn IO-aware kernels and how to manage KV cache to keep batch sizes high under load.

**Key topics**
- **FlashAttention / FlashAttention-2** (IO-aware, tiled attention; better work partitioning)
- **PagedAttention (vLLM)**: virtual-memory-like KV pages; near-zero waste
- **KV Cache Quantization (FP8)**: reduce memory footprint, increase batch size and throughput

**Free resources**
- FlashAttention (paper): https://arxiv.org/abs/2205.14135  
- FlashAttention-2 (paper): https://arxiv.org/abs/2307.08691  
- PagedAttention & vLLM (paper): https://arxiv.org/abs/2309.06180  
- vLLM FP8 **KV cache**: https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache.html

---

## 3) Decoding for Throughput & Quality

**Brief**: Modern servers win with *decoding algorithms*, not just faster kernels. Draft-then-verify and multi-head drafting can 1.5–3× tokens/s depending on model and hardware.

**Key topics**
- **Speculative Decoding** (draft with a small model; verify with the target model)
- **EAGLE / Lookahead / ReDrafter / Medusa** (parallel heads / early exits / tree drafting)
- **Batching-aware decoding** (continuous batching in vLLM/TRT-LLM)

**Free resources**
- Speculative Decoding (OpenAI): https://arxiv.org/abs/2302.01318  
- EAGLE (paper): https://arxiv.org/abs/2309.08168  
- Lookahead decoding: https://arxiv.org/abs/2307.08691 (see related work)  
- TensorRT-LLM docs (Medusa/ReDrafter/Lookahead/Eagle support): https://github.com/triton-inference-server/tensorrtllm_backend

---

## 4) Quantization (Weights, Activations, KV Cache)

**Brief**: The fastest wins come from quantization—properly. Combine **weight-only** (W4/W8) with **A8** and **KV FP8** to keep accuracy while unlocking large batch sizes.

**Key topics**
- **LLM.int8()** (vector-wise outlier handling for INT8 matmuls)
- **QLoRA** (4-bit NF4 + LoRA; train 65B on 48GB GPUs)
- **GPTQ / AWQ / SmoothQuant** (weight- or activation-aware post-training quantization)
- **KV FP8** (E4M3/E5M2) in vLLM/TensorRT-LLM

**Free resources**
- LLM.int8(): https://arxiv.org/abs/2208.07339  
- QLoRA: https://arxiv.org/abs/2305.14314  
- GPTQ: https://arxiv.org/abs/2210.17323  
- AWQ: https://arxiv.org/abs/2306.00978  
- SmoothQuant: https://arxiv.org/abs/2211.10438  
- vLLM FP8 KV cache: https://docs.vllm.ai/en/v0.6.5/quantization/fp8_e4m3_kvcache.html

---

## 5) Sparsity & Pruning

**Brief**: Prune for **speed** or **capacity**. Unstructured pruning (SparseGPT) is flexible; structured (2:4 on Ampere+) unlocks hardware speedups.

**Key topics**
- **SparseGPT** (one-shot pruning for LLMs)
- **2:4 structured sparsity** (accelerated on Ampere Tensor Cores)
- **LoRAPrune** (combine PEFT with structured pruning)

**Free resources**
- SparseGPT: https://arxiv.org/abs/2301.00774  
- NVIDIA 2:4 structured sparsity (blog): https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/  
- LoRAPrune: https://arxiv.org/abs/2305.18403

---

## 6) Long-Context Methods

**Brief**: Train short, **serve long** using positional tricks and cache policies.

**Key topics**
- **ALiBi** (train short, test long without extra params)
- **RoPE** (rotary positions) and **NTK/YaRN scaling** (extend context without retraining)
- **Prefix/Sliding caches; Chunked context**

**Free resources**
- ALiBi: https://arxiv.org/abs/2108.12409  
- RoPE: https://arxiv.org/abs/2104.09864  
- NTK-aware/YaRN scaling: https://arxiv.org/abs/2306.15595 , https://arxiv.org/abs/2309.00071

---

## 7) Streaming & State Space Models

**Brief**: For real-time or streaming, manage caches and consider **SSMs** as transformer complements.

**Key topics**
- **StreamingLLM** (finite cache with token eviction)
- **Mamba / Mamba-2** (selective SSMs for long sequences; low-latency inference)

**Free resources**
- StreamingLLM: https://arxiv.org/abs/2309.17453  
- Mamba: https://arxiv.org/abs/2312.00752  
- Mamba-2: https://arxiv.org/abs/2405.21060

---

## 8) PEFT: LoRA & Friends

**Brief**: Fine-tune **cheaply** without touching base weights; newer variants improve stability and quality.

**Key topics**
- **LoRA** (low-rank adapters)
- **DoRA** (weight decomposition for stability)
- **AdaLoRA / LoRA+** (dynamic rank, better scaling)

**Free resources**
- LoRA: https://arxiv.org/abs/2106.09685  
- DoRA: https://arxiv.org/abs/2402.09353  
- AdaLoRA: https://arxiv.org/pdf/2303.10512  
- LoRA+: https://arxiv.org/abs/2402.12354

---

## 9) Training at Scale (FSDP, ZeRO, Checkpointing)

**Brief**: Train beyond single-GPU memory with sharding and activation recomputation.

**Key topics**
- **FSDP** (parameter/grad/optimizer sharding in PyTorch)
- **DeepSpeed ZeRO** (+ offload; 3D parallelism with tensor/pipeline/data)
- **Gradient checkpointing** (sublinear activation memory)

**Free resources**
- PyTorch FSDP: https://docs.pytorch.org/docs/stable/fsdp.html  
- DeepSpeed ZeRO & Offload: https://www.deepspeed.ai/2021/03/07/zero3-offload.html  
- Checkpointing (Chen et al.): https://arxiv.org/abs/1604.06174

---

## 10) Serving Stacks & Compilers

**Brief**: Choose your battle station. Pair a serving engine with compiler/runtime optimizations.

**Stacks**
- **vLLM** (PagedAttention, continuous batching, FP8 KV, GPTQ/AWQ): https://nm-vllm.readthedocs.io/  
- **TensorRT-LLM** (speculative decoding, FP8, scheduling): https://nvidia.github.io/TensorRT-LLM/  
- **FasterTransformer** (Tensor Core-optimized kernels): https://github.com/NVIDIA/FasterTransformer  
- **llama.cpp** (CPU-first, GGUF, low-RAM): https://github.com/ggerganov/llama.cpp

**Compilers & runtime**
- **torch.compile / TorchInductor** (GPU kernels via Triton): https://pytorch.org/get-started/pytorch-2-x/  
- **CUDA Graphs** (cut Python/launch overheads): https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/

---

## 11) Advanced Retrieval & RAG

**Brief**: Go beyond naive dense retrieval.

**Key topics**
- **ColBERTv2** (late interaction, scalable reranking)
- **SPLADE** (sparse lexical expansion; hybrid retrieval with dense)
- **ColPali / multi-modal retrievers** (if you have images/PDFs)

**Free resources**
- ColBERTv2: https://arxiv.org/abs/2110.11386  
- SPLADE: https://arxiv.org/abs/2010.02666  
- ColPali: https://arxiv.org/abs/2407.01449

---

## 12) Evaluation & Benchmarking

**Brief**: Mix **task metrics** with **human/LLM preference** and **risk** metrics.

**Key topics**
- **HELM** (multi-metric, many scenarios): https://arxiv.org/pdf/2211.09110  
- **MT-Bench / Chatbot Arena** (LLM-as-a-judge + crowdsourced Elo): https://lmsys.org/blog/2023-05-03-arena/  
- **DeepEval / Confident AI** (open-source eval framework): https://www.confident-ai.com/

---

## 13) Production Tips & Checklists

**Throughput & latency**
- Turn on **continuous batching**; set max active requests per GPU engine
- Prefer **FlashAttention** kernels and enable **CUDA graphs** for hot paths
- Use **KV cache paging** and **FP8 KV** on Hopper/Ada/Blackwell for larger batches
- Right-size **prefill** vs **decode** worker pools; prioritize long prompts differently

**Stability & quality**
- Guard **max_new_tokens**, **max_time**, temperature/top-p defaults
- Pin model & tokenizer versions; seed doesn’t guarantee determinism across compilers
- Use **A/B canaries** and fallback routes (smaller model or cached answer)

**Observability**
- Log **tps**, **ttft**, **ttft_p95**, **batch size distribution**, **cache hit rate**, **OOMs**
- Emit decoding params per request; track eval scores (task + preference)

**Cost control**
- Quantize weights (W4/W8), enable **KV FP8**, and use **speculative decoding**
- Offload rarely-used models to CPU or cold GPUs; warmup on schedule
- Add **semantic caching** (e.g., request/answer cache) to avoid re-compute

**Safety**
- Add content filters and jailbreak detectors **before** model call if needed
- Store **minimal** user data; rotate logs with PII scrubbing

---

## 14) Suggested Learning Path

**Phase 1: Systems bedrock (1–2 weeks)**
- CUDA guide (memory, occupancy), Triton tutorials, FlashAttention-2

**Phase 2: Serving & decoding (1–2 weeks)**
- vLLM / TensorRT-LLM end-to-end; enable continuous batching + speculative decoding

**Phase 3: Compression (1–2 weeks)**
- QLoRA training; GPTQ/AWQ weight-only; KV FP8 in serving

**Phase 4: Long context & streaming (1 week)**
- ALiBi/RoPE scaling; StreamingLLM cache policies

**Phase 5: Eval & hardening (ongoing)**
- HELM-style metrics + MT-Bench preference; canary deploys & dashboards

---

## Bonus: Tokenization Internals (for completeness)

- **BPE**: https://aclanthology.org/P16-1162.pdf  
- **SentencePiece / Unigram LM**: https://aclanthology.org/D18-2012/ , https://arxiv.org/pdf/1804.10959

---

### Contributing

Issues/PRs welcome. Keep it **vendor-agnostic**, reproducible, and focused on **free** resources.

