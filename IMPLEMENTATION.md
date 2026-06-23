# Z-Image (Tongyi) — MLX/Swift implementation spec

Reference-grounded architecture map for porting Z-Image to Swift + MLX. Sources: the official
[`Tongyi-MAI/Z-Image`](https://github.com/Tongyi-MAI/Z-Image) modeling code, the HF model configs
(`Tongyi-MAI/Z-Image-Turbo`, community MLX repo `deepsweet/Z-Image-Turbo-6B-MLX-Q4`).

> **Status:** `ZImageConfig` + the S3-DiT building-block modules (`ZImageFeedForward`,
> `ZImageAttention`, `ZImageTransformerBlock`) are implemented and compile against MLXNN. The
> full forward (denoiser assembly, Qwen3-4B encoder, VAE) is the remaining work — and its
> **numerics must be validated on a GPU against the Python reference**; nothing here has run.

## Components & sizes

### S3-DiT transformer (single-stream)
- `dim = 3840`, `heads = 30`, `headDim = 128`, `latentChannels = 16` (VAE 4ch × 2×2 patch),
  `patch = 2`, FFN hidden `= int(dim/3*8) = 10240` (SwiGLU `w2(silu(w1 x) * w3 x)`, no bias).
- 30 main `layers` + 2 `noise_refiner` + 2 `context_refiner` blocks.
- RMSNorm (eps 1e-5). QK-norm: RMSNorm per head (dim 128) on Q and K before attention.
- RoPE: theta 256, 3D axes `dims=[32,48,48]`, `lens=[1536,512,512]`.
- AdaLN: `adaLN_modulation.0` Linear `dim → 4*dim` from `silu(t_emb)` → (scale/gate)×(attn/ffn).
  context_refiner has **no** AdaLN.
- `t_embedder` (sinusoidal 256 → mlp → dim), `cap_embedder` (RMSNorm + Linear 2560 → dim),
  `all_x_embedder` (patch embed), `all_final_layer` (norm + Linear + AdaLN → unpatchify),
  `x_pad_token` / `cap_pad_token` parameters.

### Text encoder — Qwen3-4B (must be reimplemented in MLX; swift-transformers is CoreML-only and
can't expose hidden states — use it only for the `Qwen2Tokenizer` + chat template)
- 36 layers, hidden 2560, 32 heads / 8 KV (GQA), headDim 128, FFN 9728 (SwiGLU), RMSNorm 1e-6,
  RoPE theta 1e6. Extract the **second-to-last** hidden state (layer index -2), dim 2560,
  max seq 512, chat template with `enable_thinking=True`.

### VAE (AutoencoderKL, standard SD-style)
- latent 4ch, 8× downsample, scale `0.18215`, block channels `[128,256,512,512]`, 2 resnets/block,
  attention in the mid block, GroupNorm 32.

### Scheduler
- Flow-match Euler, **shift 3.0**, default **8 steps** (Turbo). Already implemented as
  `FlowMatchEulerSampler(shift: 3.0)` in swift-diffusion-core.

## Weight keys (PyTorch state_dict; safetensors keys match exactly)
Transformer: `layers.{0-29}.attention.{to_q,to_k,to_v,to_out.0,norm_q,norm_k}`,
`layers.{N}.feed_forward.{w1,w2,w3}`, `layers.{N}.{attention_norm1,attention_norm2,ffn_norm1,ffn_norm2}`,
`layers.{N}.adaLN_modulation.0`, `noise_refiner.{0-1}.*`, `context_refiner.{0-1}.*`,
`t_embedder.mlp.{0,1}`, `cap_embedder.{0,1}`, `all_x_embedder.2-1.*`, `all_final_layer.2-1.*`,
`x_pad_token`, `cap_pad_token`. (Note `to_out.0` → the weight loader maps to the single Linear.)
Qwen3: `model.embed_tokens`, `model.layers.{0-35}.self_attn.{q,k,v,o}_proj` + `{q,k}_norm`,
`model.layers.{N}.mlp.{gate,up,down}_proj`, `{input,post_attention}_layernorm`, `model.norm`.
VAE: `encoder.*`, `decoder.*`, `quant_conv`, `post_quant_conv` (standard diffusers AutoencoderKL).

## Remaining work (in order)
1. **Qwen3-4B encoder** in MLX (36-layer GQA transformer, 4-bit `QuantizedLinear`), returning the
   layer[-2] hidden states. Tokenizer via swift-transformers `Qwen2Tokenizer` + the chat template.
2. **Denoiser assembly**: patch-embed latents, build the single-stream sequence (caption tokens +
   image tokens), 3D RoPE, run the 30+4 blocks with AdaLN from the timestep embedding, then
   `all_final_layer` + unpatchify in `unembed`. Conform to `Denoiser` (embed/blocks/unembed).
3. **VAE** decode (and encode for img2img) → wire `decode`/`initialLatent`.
4. **Weight loading**: map the safetensors keys above into the MLXNN module tree (handle the
   `to_out.0` and the quantized layout); a `RangedFileWeightSource` for streaming on iPhone.
5. **Numerical parity**: validate each stage against the Python reference on a GPU machine.
