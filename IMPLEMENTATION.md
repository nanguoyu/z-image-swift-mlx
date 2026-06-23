# Z-Image (Tongyi) — MLX/Swift implementation spec

Reference-grounded architecture map for porting Z-Image to Swift + MLX. Sources: the official
[`Tongyi-MAI/Z-Image`](https://github.com/Tongyi-MAI/Z-Image) modeling code, the HF model configs
(`Tongyi-MAI/Z-Image-Turbo`, community MLX repo `deepsweet/Z-Image-Turbo-6B-MLX-Q4`).

> **Status (key-aligned & compile-verified; numerics unvalidated):** the full `ZImageArchitecture`
> is wired and compiles — all four `DiffusionArchitecture` seam methods: `encode` (Qwen3-4B encoder
> + `Tokenizers` chat template), `makeDenoiser` (the S3-DiT `ZImageDenoiser`), `initialLatent`
> (seeded Gaussian noise), and `decode`/`encode` (the `ZImageVAE` AutoencoderKL).
>
> **Module keys verified against the reference checkpoint** (`deepsweet/Z-Image-Turbo-6B-MLX-Q4`)
> via an offline key-diff harness that flattens each MLX module tree and diffs the layer-paths
> against the checkpoint `index.json` (no weights downloaded): **transformer 483/483, VAE 122/122,
> text-encoder 398/399** — the single miss is `rotary_emb.inv_freq`, a recomputable RoPE buffer with
> no learnable params (MLX regenerates it; the loader drops it). The loader (`ZImageWeights.load`)
> quantizes the 4-bit Linears, transposes conv weights OIHW→OHWI, and filters to module
> destinations, so the three trees load exactly.
>
> **Forward shape-consistency verified** (MLX lazy graph; shapes inferred without `eval`, so the 6B
> params are never materialized): `embed → 30 blocks → unembed` round-trips a `[1,16,16,16]` latent,
> and VAE `decode`/`encode` shape correctly. This catches forward dim bugs the path diff can't.
>
> **Audit applied** (adversarial review vs the official Tongyi-MAI/Z-Image code + the authoritative
> HF configs): Z-Image uses the **FLUX VAE** (latent **16** ch, scale **0.3611**, shift **0.1159**,
> no quant convs) — not the SD VAE; DiT `in_channels=16`, `patch_size=2` ⇒ patch dim **64** (the
> x-embedder is `Linear(64→3840)`); patches are channels-**last** `(p1,p2,C)`; the single stream is
> `[image ; caption]`; the final layer's AdaLN is **scale-only** (`1+adaLN(t)`, dim→dim, no shift);
> timestep is scaled by **t_scale=1000** before embedding; the VAE downsample uses diffusers'
> asymmetric `(0,1,0,1)` pad. n_heads=n_kv_heads=30 (DiT is full MHA, not GQA).
>
> **Remaining (all need a GPU / the ~8 GB download):** the real 3D-axes RoPE (1D placeholder now);
> resolving the tokenizer + weights from the downloaded model folder rather than the hub id; and
> **numeric parity vs the Python reference** (the key-diff gate checks structure, not values —
> shapes, the parameter-free final norm, AdaLN math, and the refiner flow are still unvalidated).
> Nothing has run end-to-end.

## Components & sizes

### S3-DiT transformer (single-stream)
- `dim = 3840`, `heads = 30` (`n_kv_heads = 30` ⇒ full MHA), `headDim = 128`, `in_channels = 16`
  (FLUX-VAE latent), `patch = 2` ⇒ patch dim `16×2×2 = 64`, FFN hidden `= int(dim/3*8) = 10240`
  (SwiGLU `w2(silu(w1 x) * w3 x)`, no bias). `t_scale = 1000` (timestep ×1000 before embedding).
- 30 main `layers` + 2 `noise_refiner` + 2 `context_refiner` blocks.
- RMSNorm (eps 1e-5). QK-norm: RMSNorm per head (dim 128) on Q and K before attention.
- RoPE: theta 256, 3D axes `dims=[32,48,48]`, `lens=[1536,512,512]`.
- AdaLN: `adaLN_modulation.0` Linear `dim → 4*dim` from `silu(t_emb)` → (scale/gate)×(attn/ffn).
  context_refiner has **no** AdaLN.
- `t_embedder` (sinusoidal 256 → `linear1` → silu → `linear2` → dim), `cap_embedder` (RMSNorm +
  Linear 2560 → dim), `all_x_embedder.2-1` (patch-embed Linear), `all_final_layer.2-1`
  (parameter-free norm + `linear` + `adaLN_modulation.0` → unpatchify), `x_pad_token` /
  `cap_pad_token` parameters.

### Text encoder — Qwen3-4B (must be reimplemented in MLX; swift-transformers is CoreML-only and
can't expose hidden states — use it only for the `Qwen2Tokenizer` + chat template)
- 36 layers, hidden 2560, 32 heads / 8 KV (GQA), headDim 128, FFN 9728 (SwiGLU), RMSNorm 1e-6,
  RoPE theta 1e6. Extract the **second-to-last** hidden state (layer index -2), dim 2560,
  max seq 512, chat template with `enable_thinking=True`.

### VAE (AutoencoderKL, FLUX-family)
- latent **16ch**, 8× downsample, scale `0.3611`, shift `0.1159`, **no** quant/post-quant conv,
  block channels `[128,256,512,512]`, 2 resnets/block (decoder 3), attention in the mid block,
  GroupNorm 32. Decode: `latent/scale + shift`; encode: `(mean − shift)·scale`.

### Scheduler
- Flow-match Euler, **shift 3.0**, default **8 steps** (Turbo). Already implemented as
  `FlowMatchEulerSampler(shift: 3.0)` in swift-diffusion-core.

## Weight keys (verified against the checkpoint via the key-diff harness)
Transformer (483): `layers.{0-29}.attention.{to_q,to_k,to_v,to_out.0,norm_q,norm_k}`,
`layers.{N}.feed_forward.{w1,w2,w3}`, `layers.{N}.{attention_norm1,attention_norm2,ffn_norm1,ffn_norm2}`,
`layers.{N}.adaLN_modulation.0`, `noise_refiner.{0-1}.*` (with AdaLN), `context_refiner.{0-1}.*` (no
AdaLN), `t_embedder.{linear1,linear2}`, `cap_embedder.{0,1}`, `all_x_embedder.2-1`,
`all_final_layer.2-1.{linear,adaLN_modulation.0}` (norm is parameter-free — no `norm_final` key),
`x_pad_token`, `cap_pad_token`. (`to_out.0`/`adaLN_modulation.0` are 1-element `[Linear]` lists.)
Qwen3 (398 + recomputable `rotary_emb.inv_freq`): the converted checkpoint **strips the HF `model.`
prefix** → `embed_tokens`, `layers.{0-35}.self_attn.{q,k,v,o}_proj` + `{q,k}_norm`,
`layers.{N}.mlp.{gate,up,down}_proj`, `{input,post_attention}_layernorm`, `norm`.
VAE (122): `encoder.*` + `decoder.*` only — **no `quant_conv`/`post_quant_conv`** in this checkpoint.
The converter wraps top convs/norm asymmetrically: decoder `conv_in.conv`/`conv_out.conv`, encoder
`conv_in.conv2d`/`conv_out.conv2d`, both `conv_norm_out.norm`; resnet convs/norms are direct
(`conv1`,`conv2`,`norm1`,`norm2`,`conv_shortcut`); mid block `resnets.{0,1}` + `attentions.0`
(`to_out.0`); up/down samplers `.conv`.

## Remaining work (in order)
1. ✅ **Qwen3-4B encoder** in MLX (36-layer GQA, returns layer[-2] hidden states; keys aligned).
   Tokenizer via swift-transformers `Tokenizers` + the chat template.
2. ✅ **Denoiser assembly**: patch-embed, single-stream sequence (caption ‖ image), 30 main blocks
   as streamable blocks + noise/context refiners in `embed`, `all_final_layer` + unpatchify in
   `unembed`. Conforms to `Denoiser`. *(3D RoPE is still a 1D placeholder.)*
3. ✅ **VAE** full AutoencoderKL (encoder + decoder) → `decode`/`encode` wired.
4. ✅ **Weight loading**: `ZImageWeights.load` quantizes the 4-bit Linears, transposes conv weights,
   filters to module destinations, and updates the tree. Keys verified (see status). *(Still TODO:
   `RangedFileWeightSource` for iPhone streaming; resolve tokenizer/weights from the model folder.)*
5. **3D-axes RoPE** (theta 256, dims `[32,48,48]`) — replace the 1D placeholder.
6. **Numerical parity**: download the ~8 GB checkpoint and validate each stage (shapes, the
   parameter-free final norm, AdaLN, the refiner flow, the VAE scale factor) against the Python
   reference on a GPU. **The key-diff gate proves structure, not values — nothing has run.**
