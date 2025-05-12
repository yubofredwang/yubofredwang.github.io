---
title: Speculative Decoding, EAGLE3 and Sglang
date: 2025-05-12 10:33:00 -0700
categories: [LLM]
tags: [inference, sglang]
author: yubowang
description: A code walkthrough of of speculative decoding algorithm EAGLE3's implementation in Sglang.
math: true
---

Recently, I played around with Sglang to serve LLM models. By enabling speculative decoding with just a few changes to server arguments, I was able to increase the throughput by almost 2x. I used EAGLE3 as it is the SOTA speculative decoding approach currently. There can be even more throughput gains with careful tuning claimed in the EAGLE3 paper:

Method | Throughput (bs=1)
--- | ---
SGLang (w/o speculative, 1x H100) | 158.34 tokens/s
SGLang + EAGLE-2 (1x H100) | 244.10 tokens/s
SGLang + EAGLE-3 (1x H100) | 373.25 tokens/s

I am really curious in how this is done and spent some time to walk through the code to understand it end to end. In this post, I will introduce speculative decoding and each versions of EAGLE as EAGLE3 is the SOTA currently. Then mainly focus on the code walkthrough of the implementations in Sglang.

### Speculative Decoding

> [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192)

Speculative decoding is a technique used to accelerate the inference process. The concept is simple, we use a smaller model to "draft" a few candidate tokens and use the target model we intend to serve to "verify" if the tokens should be rejected or accepted. "Draft" here means running multiple forward passes with the smaller model. "Verify" means running one forward pass with the target model over all the candidate tokens. Since all tokens go through forward pass at the same time, this verify step is done in parallel.

Given the probability of token `x` from target model is `target_probs`, and probability from draft model is `draft_probs`. The pseudocode of the sampling algorithm when verifying looks like:

```python
output_tokens = []
while x = next_token():
    # Accept token if either:
    # 1. Target model has higher probability over the token
    # 2. Target model has lower probability but the difference is small.
    if target_probs[x] >= draft_probs[x] or target_probs[x] / draft_probs[x] <= sample_uniform(0, 1):
        tokens.append(x)
    else:
        # Resample to guarantee always a new token is generated.
        p_new = norm(max(0, target_probs[x] - draft_probs[x]))
        sample x from p_new
        output_tokens.append(x)
        break
return output_tokens
```

<i> We won't cover details of original speculative decoding algorithm here since it is not the focus of this post. Please refer to the original paper for more details. The actual implementation in Sglang uses a slightly different sampling method. </i>

### EAGLE, EAGLE2 and EAGLE3

#### EAGLE

> [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/pdf/2401.15077)

Unlike traditional speculative decoding where the draft model generates tokens autoregressively based solely on previous tokens, EAGLE introduces a novel approach. It leverages the features (f in the diagram) from the second-to-top layer (before the LM head), e.g the last hidden states, and concatenates them with token embeddings (e in the diagram) to generate the next token in the draft model. The process works as follows:

Assuming input sequence is "How can"

1. The target model performs a forward pass on the input sequence, generating a token t1 ("I") and its hidden states f0 ("how can").
2. The draft model passes all the input sequence plus the token t1 through the embedding layer (borrowed from the target model) to produce token embedding e1 ("can I").
3. In the first draft step, the token embedding e1 ("can I") is concatenated with hidden states f0("how can") and fed into the AutoRegression Head (FC layer + 1 Decoder Layer) of the draft model. This produces new hidden states f1 ("make"). Note that all prefix tokens (like "How" in the diagram) are processed similarly in this first draft step in parallel.
4. The hidden states f1 ("make") from 3 are passed through the LM head (borrowed from the target model) to obtain a probability distribution. A new token t2 ("make") is sampled from this distribution, and the hidden states f1 ("make") are preserved for the next draft step. When using topk sampling, k tokens are sampled instead of just one.
5. In subsequent draft steps, t2 ("make") and f1 ("I") are fed into the draft model, and the process repeats. If k tokens were sampled in the previous step, they are processed in parallel in the next forward pass.

This approach generates a tree structure of tokens where the tree depth equals the number of forward steps. Since EAGLE's static tree approach has been superseded, we can look directly at EAGLE2 for an illustration of the dynamic tree structure.

<img src="assets/eagle.png" alt="Eagle Model" width="80%"/>


#### EAGLE2

> [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)

There is no model architecture change in EAGLE2. EAGLE2 introduced a dynamic draft tree which ranks the drafted tokens. During expansion phase, we select the top-k nodes with the highest confidence scores as the input to the next draft forward step. During rerank phase, we rerank all draft tokens and select the a top number of m tokens with the highest confidence scores. During the verification phase, target model verifies the draft tree. Therefore, an attention mask must be adjusted according to the tree structure to ensure that each token can only attend to its ancestor nodes.

The diagram is using topk = 2 and num_steps = 3. 


<img src="assets/eagle2.png" alt="Eagle2 Model" width="80%"/>

<img src="assets/attention_mask.png" alt="Attention Mask" width="80%"/>


#### EAGLE3

> [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)

EAGLE3 has a different model architechture. Instead of using the last hidden states of target model, EAGLE3 uses 3 hidden states: low, mid and high layers in the model. The three hidden states are concatenated and passed through a FC layer to generate g(instead of f). Then this g is concatenated with token embedding the same way as EAGLE. After the first step, the auxiliary information(a in the diagram), e.g hidden states from the last draft model forward pass is concatenated with the embedding sequence and fed into the draft model again. Another important difference is that EAGLE3 uses it's own trained LM head unlike EAGLE taking the LM head from the target model.

I think this change is because only the last hidden states are not enough to capture the information similar enough to the target model as the model has deep layers.

<img src="assets/eagle3.png" alt="Eagle3 Model" width="80%"/>


### Sglang Code Walkthrough

#### Server Args

Three server args that user can tune when enabling eagle3 in sglang are important for understanding the process and are mentioned a few times in this walkthrough:
- speculative_num_steps: How many draft forward passes we run before verifying.
- speculative_num_draft_tokens: The number of tokens proposed in a draft.
- speculative_topk: Each draft pass will generate topk tokens.

A sample command:

```bash
python3 -m sglang.launch_server --model Meta-Llama-3.1-8B-Instruct --speculative-algorithm EAGLE3 --speculative-draft-model-path sglang-EAGLE3-Llama-3.1-Instruct-8B --speculative-num-steps 5 --speculative-eagle-topk 4 --speculative-num-draft-tokens 16 --mem-fraction 0.7 --dtype float16 --trust-remote-code
```

Note: If you want to draft 3 tokens and draw top 2 tokens for each draft step, you only need 1 draft pass. Because the forward extend from last round will always provide 1 more token.

#### Model Code

Let's use LlamaForCausalLMEagle as an example to illustrate the model architecture. The model contains three main components:
- Embedding layer: This layer convert a token into a token embedding sequence. The embedding generated will be concatenated with the hidden states.
- AutoRegression Head: This layer consists of one decoder layer and a fully connected layer. The layer is of dimension (bs, seq_len, 2*hidden_size). It takes the concatenated token embedding and hidden states as input and generates a hidden states "f" of dimension (bs, seq_len, hidden_size).
- LM Head: This layer takes the hidden states "f" and generates the final output probability distribution.


```python
# Code modified for illustration purpose
class LlamaForCausalLMEagle(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        ...
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        # Most of the EAGLE model has num_hidden_layers = 1
        self.layer = LlamaDecoderLayer(config)
        self.fc = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        ...
    ) -> torch.Tensor:
        # embed >> fc >> decoder
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.fc(torch.cat((hidden_states, forward_batch.spec_info.hidden_states), dim=-1))
        residual = None
        hidden_states, residual = self.layer(positions, hidden_states, forward_batch, residual)
        return hidden_states + residual
```

During eagle worker initialization, `draft_model_runner.model.set_embed_and_head(embed, head)` will be called to set the weights of the embedding layer and LM head from the target model.

EAGLE3 model is slightly different:
1. lm_head's weights are loaded from the draft model instead of taking from the target model. Thus, during eagle3 worker intialization, only embedding layer is set by `draft_model_runner.model.set_embed(embed)`.
2. Eagle3 also sets `capture_aux_hidden_states=True` which means the hidden states produced from the decoder layer are kept during the draft. `set_eagle3_layers_to_capture()` will be triggered on the target model with `layers_to_capture = [2, num_layers // 2, num_layers - 3].` which corresponds to low, mid and high.
3. First draft step uses the target model's hidden states, which is a concatenation of 3 hidden states. The following draft steps use the aux hidden states captured from the previous draft step.
4. The model now goes as fc(first step) or aux-> decoder(captures aux for next step) -> norm -> logits_processor(lm_head) -> sampler -> tokens.

```python
# Code modified for illustration purpose
class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        # override qkv with 2 * hidden_size, because we are concatenating the token embedding and hidden states
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)
        # Concatenate the token embedding and hidden states
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        # Self Attention + Layer Norm + MLP
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, forward_batch=forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        # fc -> decoder -> norm
        self.fc = torch.nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.midlayer = LlamaDecoderLayer(config, 0, quant_config, prefix)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LlamaForCausalLMEagle3(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(config.draft_vocab_size, config.hidden_size,)
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = True

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        hidden_states = forward_batch.spec_info.hidden_states
        # First draft step uses the target model hidden states, which is a concatenation of 3 hidden states
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)
        # Otherwise the aux hidden states are used
        residual = None
        hidden_states, residual = self.midlayer(...)
        hidden_states_to_logits, hidden_states_to_aux = self.norm(hidden_states, residual)
        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]
```

#### Eagle Worker

Majority of the code locates in two files in speculative folder:
- python/sglang/srt/speculative/eagle_worker.py - defines a EAGLEWorker class which is responsible for batch generations of tokens.
- python/sglang/srt/speculative/eagle_utils.py - defines classes EagleDraftInput, EagleVerifyInput.

Important functions are listed below:

```
eagle_worker.py
--| EAGLEWorker
----| forward_batch_speculative_generation
----| forward_target_extend
----| forward_draft_extend
----| forward_draft_extend_after_decode
----| draft
----| draft_forward
----| verify

eagle_utils.py
--| EagleVerifyInput
----| prepare_for_verify
----| verify
----| build_tree_kernel_efficient
--| EagleDraftInput
----| prepare_for_extend
----| prepare_extend_after_decode
```

EAGLEWorker is intialized in scheduler. Target worker is passed into draft worker as a parameter:

```python
# Code copied from scheduler.py
# Regular Tensor Parallel worker for target model
self.tp_worker = TpWorkerClass(server_args=server_args)

# Launch a draft worker for speculative decoding
if self.spec_algorithm.is_eagle():
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

    self.draft_worker = EAGLEWorker(
        server_args=server_args,
        target_worker=self.tp_worker,
    )
```

<img src="assets/eagle_worker.png" alt="Eagle Worker" width="80%"/>

<i>Yellow rectangles are forward #1 from EAGLE diagram. Green recetangles are the rest of the forwards.</i>

When a batch of requests comes in, `forward_batch_speculative_generation` is triggered in `batch_schedule.py` as the entry point for the process. In this function, the following steps are performed:

1. forward_target_extend: Forward with the target model and get hidden states and returns `(logits_output, next_token_ids)`. This step generates 1 token.
    - This forward will generate a token and the hidden states. The worker batch for target model's `caputre_hidden_mode` is set to `CaptureHiddenMode.FULL` in order capture the hidden states from the target model. As we have mentioned above, there are 3 layers of hidden states is captured for EAGLE3.
    - Because of the nature of model forward, one token is generated and will be used as the root node for the drafting step.
2. forward_draft_extend: Forward with the draft model and get hidden states.
    - Creates `EagleDraftInput(hidden_states=logits_output.hidden_states, verified_id=next_token_ids)`. Because the `next_token_ids` are generated from the target model, it is considered as verified tokens. We call `prepare_for_extend` to concatenate `input_ids[1:]` with verified_ids(generated 1 token from target) for draft model. This is because we need to combine hidden states of token f0...fn and tokens t1...t(n+1) in the draft model.
    - Then we run` draft_model_runner.forward(forward_batch)`, this will generate logits_output where we can sample tokens from. This actually corresponds to the first forward pass described in the EAGLE paper. The tokens are sampled with topk in `capture_for_decode(logits_output)`.
    - One additional thing to notice is before we run draft forward, we set capture_hidden_mode to `CaptureHiddenMode.LAST` as only the last hidden states from the draft model is needed for the next draft step.
3. draft: Run draft forward pass for `speculative_num_steps - 1` times(1 forward already runs during draft extend) to generate (scores, tokens, parents).
    - Calls `draft_forward(forward_batch)`, this step will keep calling `select_top_k_tokens` and  `draft_model_runner.model.forward` alternately. topk_p from forwrad will be used to select_top_k_tokens.
        ```python
        for i in range(self.speculative_num_steps):
            # First step after extend will use topk_p as score while the rest will calculate scores based on parent nodes as well.
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            if i == self.speculative_num_steps - 1:
                break
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        ```
    - By the end of the draft forward, a draft tree is generated. EagleVerifyInput is created with score_list, token_list and parent_list which are necessary information for reranking.
4. rerank: Though rerank is only introduced in EAGLE2, it is used for all EAGLE speculative decoding in SGLang. rerank takes the scores and the tokens and generates a sorted tree of draft tokens and a tree attention mask to be used during verification. Throughout the rerank process, the root nod of the draft tree is always a verfied token from the target model.
    - Only `speculative_num_steps` number of draft tokens are kept after `EagleVerifyInput.build_tree_kernel_efficient_preprocess` triggers.
        ```python
        def build_tree_kernel_efficient_preprocess(self, verified_id, ... ,score_list, num_verify_tokens):
            score_list = torch.cat(score_list, dim=1).flatten(1)  # (b, n); n= 1 + (num_steps-1) * self.topk
            ss_token_list = torch.cat(
                token_list, dim=1
            )  # b, (self.topk + (num_steps-1) * self.topk)
            # Select only num_verify_tokens - 1 tokens because the first token is generated from the target forward.
            top_scores = torch.topk(score_list, num_verify_tokens - 1, dim=-1)
            top_scores_index = top_scores.indices
            top_scores_index = torch.sort(top_scores_index).values
            draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)
            # Concat draft tokens with verified tokens produced from target model forward, which is the root node of the tree.
            draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()
        ```
    - `build_tree_kernel_efficient` will generate an attention mask based on the draft tokens kept from preprocess. Even though the draft tokens are reranked from the prevous step, nodes from next level are always ranked lower than the nodes from parent level. Thus, the tree mask is still a sparse lower triangular matrix.
5. verify: runs target model forward with the draft tree from the draft step.
    - The first step is to get the logits_output by running forward with target model: `target_worker.forward_batch_generation` with forward_mode == ForwardMode.TARGET_VERIFY. TARGET_VERIFY mainly controls the attention backend to use the mask generated during rerank step.
    - Then the logits go through softmax, top_k_renorm_prob, top_p_renorm_prob and finally feed into `tree_speculative_sampling_target_only` where the target distribution is used to accept or reject the tokens. This process solely base on target model's probability distribution and heauristic thresholds. Draft distribution is not used currently. This process also generates a "bonus" token from the target model's probability distribution to guarantee at least one token per decoding step. Think of this as a regular forward and sampling process. We eventually have accepted_length + 1 tokens generated from this step.
    - Towards the end, we construct EagleVerifyOutput with the new `verified_id = predict[accept_index]`. This verified_id contains accepeted token ids including the bonus token.
6. forward_draft_extend_after_decode: After the verification step, certain tokens are accepted making the input for next round of draft changed. We append the verified tokens to input_ids and set `verified_id` to only keep last token before runnning the draft model forward again. This verified_id will be the new node for the next round's draft tree. Similar to forward_draft_extend, we set capture_hidden_mode to `CaptureHiddenMode.LAST` as only the last hidden states from the draft model is needed for the next draft step. We again call `capture_for_decode` to sample topk tokens from the draft model. 


#### Interaction with KVCache

During EAGLEWorker initialization, draft worker shares the same kv pool alloacator but has their own kv cache pool. My understanding is even though draft worker and target worker do not share kv cache but they are processing the same batch of tokens throughout the process and their token-to-kv pool should eventually be consistent with each other. This also helps ease up the handling of the radix cache.

```python
# In draft worker, set kv cache pool to be the same as target worker
self.req_to_token_pool, self.token_to_kv_pool_allocator = (
    target_worker.get_memory_pool()
)
```

During the draft step, multiple rounds of forward pass are run using the draft model. However, the kv cache can be discarded after the draft step is done(We don't know which tokens' kv cache to keep until verify, `forward_draft_extend_after_decode` will generate the new kv cache instead). Thus, we will keep the state of token to kv pool allocator before drafting and restore it after the draft step is done.
```python
out_cache_loc, token_to_kv_pool_state_backup = batch.alloc_token_slots(
    num_seqs * self.topk * self.speculative_num_steps, backup_state=True
)
self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)
```

Before the verify step in `prepare_for_verify`, we need to allocate the token slots for the drafted tokens in target workers to write to and store them in the req_to_token_pool. This is done by calling `assign_req_to_token_pool`. This is similar to how extending tokens work in regular worker.

```python
batch.input_ids = self.draft_token
batch.out_cache_loc = batch.alloc_token_slots(len(batch.input_ids))
assign_req_to_token_pool[(bs,)](
    batch.req_pool_indices,
    batch.req_to_token_pool.req_to_token,
    batch.seq_lens,
    end_offset,
    batch.out_cache_loc,
    batch.req_to_token_pool.req_to_token.shape[1],
    next_power_of_2(bs),
)
```


When the verify step is done, the rejected tokens' kv cache from the target model needs to be cleaned up.

```python
# Filter out accepted tokens
accept_index = accept_index[accept_index != -1]
evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
# Build maks
evict_mask[accept_index] = False
if page_size != 1:
    align_evict_mask_to_page_size[len(batch.seq_lens),](
        batch.seq_lens,
        evict_mask,
        page_size,
        self.draft_token_num,
        next_power_of_2(self.draft_token_num),
    )
# Shared kv pool allocator
token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
```

Equivalently to `prepare_for_verify`, after the verify is done, we will assign these token locations back to the req_to_token_pool for the `forward_draft_extend_after_decode` of the draft worker.

```python
assign_req_to_token_pool[(bs,)](
    batch.req_pool_indices,
    batch.req_to_token_pool.req_to_token,
    batch.seq_lens,
    batch.seq_lens + accept_length + 1,
    batch.out_cache_loc[accept_index],
    batch.req_to_token_pool.req_to_token.shape[1],
    next_power_of_2(bs),
)
```


Thanks for reading all the way through! This is for sure one of the more complicated processes and challenging code to understand. My learnings from diving into the codebase is:
1. Understand the details of algorithm/design first before diving into the code. This helps a lot in understanding the code.
2. If possible, avoid modifying variables in place. In-place modifications are hard to keep track of and the developers of other parts of the code can have false assumptions.
3. There are triton, cutlass kernels in sglang codebase. I think they can be documented better too.
4. The overall design of the worker is general enough to be extended to another speculative decoding algorithm. However, it will be more helpful if EAGLEWorker class can be abstracted out to a base class.

Also, never forget this:

> <i>"But We Don't Do Things Because They Are Easy Hm? We Do Them Because They Are Profitable." - Tom Nook </i>