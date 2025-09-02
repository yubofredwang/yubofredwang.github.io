---
title: Flex Attention and Triton, Supporting Long Context Training in SpecForge
date: 2025-05-12 10:33:00 -0700
categories: [LLM]
tags: [training, sglang, SpecForge]
author: yubowang
description: This post summarizes how we used Flex Attention and custom triton kernels to support long context training for a EAGLE3 model in SpecForge.
math: true
---

Earlier this year, I had a post on how EAGLE3 speculative decoding works in SGLang. EAGLE3 has an advantage over MTP(Multi-Token Prediction) as MTP is usually trained together during the pre-training phase. However, EAGLE3 can be trained seperately. As Linkedin fine-tune open source LLMs most of the time, EAGLE3 is a great fit for our use cases.

SGLang community has released a open source project named SpecForge, which is a framework for training speculative decoding draft models, particularly EAGLE3, to speed up inference of LLMs. We quickly adopted SpecForge into Linkedin's internal use cases such as LLM-as-a-Judge, Agentic LLMs. During the training, we encountered challenges to scale the training up in the long context scenarios. Especially for the agentic use cases, our training data can easily reach 16K or even 32K tokens in length. In this post, I will summarize how we used Flex Attention and a custom triton kernels to reduce the memory footprint and increase the speed.

The improvements are:

- [Add Flex attention backend, uses 10~20x less memory in long context training #97](https://github.com/sgl-project/SpecForge/pull/97)
- [Optimize loss calculation with in-place gradients calculation ~40% memory save #185](https://github.com/sgl-project/SpecForge/pull/185)

We successfully trained a EAGLE3 draft model for Qwen3-30B-A3B with 16K length context with our own created blend dataset.

## EAGLE3 Training

Why memory has become a bottlneck for EAGLE3 draft model training? Mainly due to two reasons:
1. EAGLE3 uses the 3 decoder layers'(low, mid, high) output hidden states generated from the target model and combine them with the token embeddings of draft model to feed into draft model. These hidden states can be generated either online/on-demand or offline where all hidden states are stored in a storage. SpecForge supports both online and offline training modes. However, offline training requires storing 3 * (batch_size * seq_len) * hidden_size of bfloat16 tensor. So for Qwen3-30B-A3B with 16K context length of batch_size 1, it will be 3 * 1 * 16384 * 2048 * 2 bytes = 201326592 bytes, and that is 201 MB alone for 1 entry. We have around 50k data. That is around 10 TB of storage needed! Also, we need to count in the overhead of reading data from FileSystem -> CPU -> GPU. Thus, online training is a more popular choice for us. Under this setting, target model coexists with the draft model during training, reducing the memory capacity that can be used for the draft model itself.
2. A part of the novelty of EAGLE3 training is its Training Time Test (TTT) mechanism. During training, draft model will run forward N times. For each step, it will generate the next logits for each token given the previous inputs. Then these logits are used as the input sequence for the next step. For each step, we calculate a loss for the new input sequence. Please notice, even if each step only generates a hidden states. We are referring it as "new tokens" for the purpose of easy understanding. The diagram below illustrates the TTT mechanism.

<img src="assets/ttt.png" alt="TTT Training" width="80%"/>


The process can be described using pseudocode as below:

```python
hidden_states = target_model(input_ids)
input_emb = target_model.input_embedding(input_ids)
losses = []
for ttt_step in range(N):
    hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
    step_hidden_states = draft_model_decoder(
        hidden_states=hidden_states,
        ...other_args,
    )
    hidden_states = step_hidden_states # update hidden states for the next step
    loss = compute_loss(step_hidden_states, target_probabilities)
    losses.append(loss)
final_loss = weighted_sum(losses) / N
final_loss.backward()
```


The TTT mechanism requires a sparse attention calculation. The new "tokens" should attend to all the previous tokens. Thus, KV cache is needed to store the tokens from previous steps.

<img src="assets/training_mask.png" alt="TTT Training" width="80%"/>


The idea in the original EAGLE3 paper is to calculate an attention weights for the first causal part and the following sparse part separately and then run softmax. Recall that:

> attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

and the attention weights can be calculated as:
> weights = QK^T / sqrt(d_k)

For the first part, the attention can be calculated with:

```python
attn_weights = query_states @ k.transpose(2, 3) / math.sqrt(head_dim)
```

The rest can be calculated with:

```python
for i in range(kv_cache_length):
    ki = kv_cache[i]
    # dot product between query and key
    attn_weightsi = (qi * ki).sum(-1) / math.sqrt(head_dim)
    # concatenate the attention weights
    attn_weights = torch.cat((attn_weights, attn_weightsi), dim=-1)

attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
attn_output = torch.matmul(attn_weights0, v0)
for i in range(1, lck):
    vi = cache_v[i]
    attn_weightsi = attn_weights[..., q_len + i - 1]
    attn_outputi = attn_weightsi[..., None] * vi
    attn_output = attn_output + attn_outputi
```

This sounds good as it reduces the amount of calculation needed. However, this approach uses naive softmax calculation which stores the entire probabilities after softmax unlike Flash Attention's softmax. According to memory profiling, we can see memory usage spike accordingly:

<img src="assets/memory_profiling_softmax.png" alt="Memory Profiling" width="80%"/>


## Flex Attention

[Flex attention](https://arxiv.org/pdf/2412.05496) is a project from pytorch team. The core idea is to leverage torch inductor to compile a simple python function into a hand-written highly efficient attention kernel implemented in triton. It is extremely useful for sparse attention calculation due to its mechanism to skip blocks that does not require calculation. FlexAttention implements a BlockMask data structure which efficiently precomputes which blocks along a dimension can be skipped. Besides, it also devides the rest of the blocks into partial blocks and full blocks. Full blocks can skip the mask_mod operation, which makes it a little bit faster with less operations. Most importantly, the generated kernel uses online softmax to save memory, which fits our need here. It also supports broadcasting on batch dimension and supports Grouped Query Attention (GQA). Thus, our job become simple: construct a BlockMask based on the TTT mechanism described above and let Flex Attention handle the rest.

We build eagle BlockMask as below:

```python
def causal_mask(b, h, q_idx, kv_idx):
    # Causal will keep shrinking by 1 diagnol due to appended suffix
    # Shirnk the causal by diagnol
    causal_mask = q_idx - shift_left >= kv_idx
    padding_mask = kv_idx < seq_lengths[b]
    return causal_mask & padding_mask

def suffix_mask(b, h, q_idx, kv_idx):
    suffix_mask = kv_idx >= Q_LEN
    padding_mask = kv_idx % Q_LEN < seq_lengths[b]
    diagnol_mask = (kv_idx - q_idx) % Q_LEN == 0
    return suffix_mask & padding_mask & diagnol_mask

mask_mod = or_masks(causal_mask, suffix_mask)
```

Then we pass the maskmod when creating block mask and pass the created block mask into flex attention.

```python
block_mask = create_block_mask_func(
    mask_mod=generate_eagle3_mask(
        seq_lengths=seq_lengths,
        Q_LEN=q_len,
        KV_LEN=key_cache.shape[-2],
        shift_left=lck,
    ),
    B=bsz,
    H=1,  # Rely on broadcast
    Q_LEN=q_len,
    KV_LEN=key_cache.shape[-2],
    device=query_states.device,
)

attn_output = flex_attention(
    query=query_states,
    key=key_cache.contiguous(),
    value=value_cache.contiguous(),
    block_mask=block_mask,
    enable_gqa=True,
)
```

You might notice that we need to pass in KV cache. We leveraged DynamicCache class defined in transformers libarary provided by huggingface to store the KV cache.

Sounds simple, right? However, during the benchmarking, we did not see much memory reduction or speedup. In some cases, the memory usage even increased. After some investigation, we found out we need torch.compile() to leverage the compiled triton kernel. However, after enabling torch.compile(dynamic=True), the outputs of the flex attention does not match the naive implementation as the input shape can change due to different length each batch. I tried a few things such as using padding to multiple of 256 and leverage StaticCache to fix the shape. However, StaticCache can not be used for backward propagation as the one big tensor allocated will be modified multiple times during the backward pass, which is not allowed. Things seem to be stuck.

Things took a turn when pyTorch team release [torch2.8](https://pytorch.org/blog/pytorch-2-8/). After upgrading to torch2.8. The kernel compiled handles dynamic shapes correctly.

Another lesson learned is that when we write test to compare naive attention with flash-attention style attention mechanism, we should avoid using torch.randn() as it does not cap on the range of the values. This can lead to overflow causing precision tests to fail.

After the integration, we can see great memory reduction and speedup during the training:

<img src="assets/benchmark.png" alt="Speed/Memory Profiling Flex Attention" width="80%"/>

## In-place Gradient Calculation

After Flex Attention optimization, the bottleneck now shifts to the loss calculation phase. EAGLE3 training calculates the soft target probabilities loss for each TTT step. The loss calculation is done as below:

```python
@torch.compile(dynamic=None)
def _compute_loss(logits, target_p, position_mask):
    logits = logits.float()
    out_logp = nn.LogSoftmax(dim=2)(logits)
    plogp = target_p * out_logp
    loss = -torch.sum(position_mask * plogp, 2).mean()
    return loss
```

See the memory profiling below. The memory peak is the TTT steps of loss calculation's gradients and intermediate tensors for backward pass.

<img src="assets/mem_profiling_loss.png" alt="Memory Profiling Loss Calculation" width="80%"/>

Thus, the second improvement we made is to optimize the loss calculation with in-place gradient calculation. This is a trick to leverage the input logits' tensor to store the gradients during the backward pass. This can roughly save half of the memory since gradients are of the same shape as the logits. This technique is used for cross entropy kernel in Liger Kernel repo. However, we can't directly leverage the library for the following reason:
1. Liger-Kernel only handles one hot encoding cross entropy loss. In SpecForge, the loss is calcuated with soft target probabilities. Due to this, the gradients calculation needs to be separated into another kernel, where Liger fuses forward and backward pass into one kernel.
2. Liger-Kernel applies kernel replacement through monkey patching. Here, we don't have a module. Instead, we create a new LogSoftmaxLoss class that extends torch.autograd.Function and use it directly.

Here is the benchmark results of the peak memory reduction:

| Config (B, T, V)   | PyTorch (ms) | Triton (ms) | Speedup | PyTorch Mem (GB) | Triton Mem (GB) | Memory Save |
|---------------------|--------------|-------------|---------|------------------|-----------------|-------------|
| (1, 1024, 32000)   | 449.08       | 435.22      | 1.03x   | 1.85             | 0.98            | 46.7%       |
| (1, 4096, 32000)   | 127.67       | 7.03        | 18.15x  | 7.32             | 5.62            | 23.3%       |
| (1, 4096, 64000)   | 20.78        | 24.35       | 0.85x   | 14.65            | 11.23           | 23.3%       |
| (1, 8192, 32000)   | 20.48        | 13.56       | 1.51x   | 21.48            | 14.65           | 31.8%       |
| (1, 8192, 64000)   | 41.14        | 48.11       | 0.86x   | 29.30            | 22.46           | 23.3%       |
| (1, 16384, 32000)  | 41.11        | 26.95       | 1.53x   | 42.97            | 29.30           | 31.8%       |


