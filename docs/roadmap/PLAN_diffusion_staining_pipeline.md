# Plan — Server-Side Generative Staining Pipeline (Approach A)

**Status:** DRAFT for review · **Owner:** (client) · **Author:** engineering · **Date:** 2026-07-07
**Supersedes:** client-side "Algorithmic Color Profiling & Canvas Texturing" (Approach B, shelved)
**Related infra:** `cloudrun_inference/`, `modal_inference/`, `fence-staining-visualizer/`

> Read this first: Section 1 (verdict) and Section 2 (open decisions). Everything
> after is the detailed design that those decisions drive. Tell me what to change
> in Section 2, or say "execute Phase 0" to start the proof-of-concept.

---

## 1. Executive summary & verdict

**Goal.** Replace the hand-coded browser recolor with a server-side generative
pipeline that re-renders the fence area as photorealistic freshly-stained wood —
correct commercial swatch color, real premium-sealer sheen, real grain and
lighting — while remaining recognizably the *customer's* fence.

**Verdict.** Diffusion (SDXL Inpaint + ControlNet) is the right tool for *texture
realism* and is worth building. But the naive "high-denoise, pre-tint, hope the
color sticks" formulation will fail the two things this product is actually sold
on: **exact stain color** and **"that's my fence."** This plan therefore adopts a
**controlled variant** as the recommended path:

- **Deterministic color + identity first, diffusion for realism second.** Build an
  exact-swatch, structure-preserving base image, then run a *low-denoise*
  ControlNet pass to add grain/sheen without moving color or geometry.
- **Deterministic color-lock last.** After diffusion, re-impose the exact swatch
  chroma so output color is guaranteed, not probabilistic.
- **Never fail the user.** If diffusion errors, times out, or scores low on the
  QA gates, fall back to the deterministic base render.

This keeps diffusion's realism ceiling while removing its two fatal risks. The
client's original aggressive variant is preserved as **Variant B** (Section 6.3)
if the low-denoise path proves too conservative in the Phase 0 bake-off.

**Non-goals (this plan).** Re-training the segmentation model; changing the color
swatch catalog; mobile on-device diffusion; video.

---

## 2. Open decisions (need client input before execution)

| # | Decision | **Chosen (client, 2026-07-07)** | Notes / implication |
|---|----------|-------------------------------|---------------------|
| D1 | Denoise philosophy | **A — low-denoise over deterministic base** | protects color & identity; Variant B kept only as a Phase-0 comparison |
| D2 | Color guarantee | **A — deterministic color-lock post-pass** | **non-negotiable** given the #1 requirement below; ΔE is the primary gate |
| D3 | Latency/UX model | **B — every color change hits GPU** | ⚠ differs from recommendation → makes caching (D7) + debounce + warm-up critical; see §7, §9 |
| D4 | Warm vs cold GPU | **B — scale-to-zero + accept cold start** | pilot economics; revisit warm pool at launch (§9). Within a session the instance stays warm after the first render |
| D5 | Base model | **SDXL-Lightning (4–8 step)** | realism + speed; SD1.5+ControlNet documented as a cost fallback |
| D6 | Host | **Cloud Run L4** | extend the existing `/detect` service with a new `/render` endpoint |
| D7 | Per-color caching | **Yes — cache by (imageHash, colorHex)** | with D3-B this is the primary cost **and** latency lever |

> **#1 HARD REQUIREMENT (client-confirmed): the rendered fence must match the exact
> commercial swatch color.** This makes the deterministic color-lock (D2-A) mandatory
> and **CIEDE2000 ΔE the primary acceptance gate** (§3). Any variant or diffusion roll
> that cannot hold ΔE ≤ 3 does not ship — it falls back to the deterministic
> exact-color render.

Decisions **locked 2026-07-07**. Sections below reflect them; **D3-B** is flagged
wherever it changes the design (§7, §9). Say the word to revise any cell.

---

## 3. Success criteria (acceptance gates)

A render "passes" only if ALL of these hold. These are the Phase-0 go/no-go metrics.

| Dimension | Metric | Target | Why |
|-----------|--------|--------|-----|
| **Color fidelity** — **PRIMARY, #1 req** | CIEDE2000 ΔE between rendered fence (median + per-board) and target swatch | **ΔE ≤ 3** (excellent), hard fail > 6 | The core product promise; gate #1 — a roll above target falls back to the deterministic exact-color render |
| **Identity** | Plank-line / edge correlation (input vs output, within mask); human "same fence?" | ≥ 0.8 edge-IoU on strong seams; ≥ 90% human yes | Before/after credibility |
| **Realism** | Human rating vs "looks like a real freshly-stained fence" (1–5) | mean ≥ 4.0, none ≤ 2 | The reason to use diffusion at all |
| **Containment** | Fraction of changed pixels outside the fence mask | ≤ 0.5% | No bleeding onto trees/sky/yard |
| **Latency (warm)** | p50 / p95 end-to-end at production res | ≤ 4s / ≤ 7s | Interactive-enough for "HD render" |
| **Latency (cold)** | first-request after scale-up | ≤ 40s (with spinner + messaging) | Bounded, communicated |
| **Reliability** | render error / hallucination rate (auto-QA + fallback triggered) | ≤ 2% reach fallback; 0% user-visible failures | Fallback must catch the rest |
| **Cost** | GPU-seconds per accepted render (warm) | ≤ 6 GPU-s | Drives unit economics (§9) |

**Test set:** 30 real customer-style photos spanning {weathered-grey, algae/green,
peeling, new unstained, mixed corner, heavy occlusion by foliage, low light,
backlit}. Curated from `data/images/` + the client's `Fence-simulator testing images`.

---

## 4. Architecture

### 4.1 Data flow (recommended path)

```
[Browser upload]
   │  multipart POST /render {image, colorHex, family, quality}
   ▼
[1 Segmentation]  DINOv3 ONNX (EXISTING /detect)  ──► fence mask (upscaled to working res)
   ▼
[2 Condition classifier]  zero-shot SigLIP/CLIP on masked fence crop  ──► {weathered | clean} profile
        (+ chroma heuristic as fail-safe fallback)
   ▼
[3 Deterministic base recolor]  intrinsic structure-preserving recolor to EXACT swatch
        (keeps planks/gaps/knots/lighting; neutralizes weathering; inpaints peel/algae)
   ▼                                   ┌─────────────────────────────┐
[4 Control signals]  depth (Depth-Anything-V2) + soft edge (HED/canny) from ORIGINAL
   ▼                                   └─────────────────────────────┘
[5 SDXL Inpaint + ControlNet]  img2img on the base render, LOW denoise (0.25–0.40),
        LCM/Lightning 4–8 steps, COLOR-NEUTRAL texture prompt, mask-restricted
   ▼
[6 Deterministic color-lock]  re-impose exact swatch chroma on diffusion luminance
   ▼
[7 Composite]  paste fence pixels back over the untouched original, feather mask edge
   ▼
[8 Auto-QA + cache]  ΔE / containment / identity checks → pass or FALLBACK to [3]
   ▼
[Return HD image]  (+ store by (imageHash,colorHex) for instant re-serve)
```

### 4.2 Why each stage exists (the guardrails)

- **[3] deterministic base** gives the diffusion a starting image that is already
  the *right color* and the *right fence*. Low-denoise diffusion then only has to
  add texture — it cannot wander to a new color or a new fence.
- **[4] depth + edge ControlNet** locks geometry: planks, posts, rails, perspective.
- **[5] low denoise (0.25–0.40)** is the single most important knob. High denoise =
  realism but color/identity drift; low denoise over a good base = realism *and*
  fidelity. Phase 0 sweeps this per profile.
- **[6] color-lock** converts "probably the right color" into "guaranteed the right
  color" — decompose the diffusion output into luminance (keep, it carries the new
  grain/sheen) + chroma (replace with the exact swatch, ΔE-measured).
- **[7] mask composite** guarantees trees/sky/yard are pixel-identical to the upload.
- **[8] auto-QA + fallback** means a bad diffusion roll never reaches the user.

### 4.3 Condition-based parameter profiles (from the client's Approach A)

| Profile | Trigger (inside mask) | denoise | ControlNet scale | Notes |
|---------|----------------------|---------|------------------|-------|
| **Weathered/algae** | **low chroma (near-grey)** primary · green-excess (algae) · σ = weak secondary | 0.35–0.40 | 0.55–0.65 | more freedom to rebuild rough surface; base render already cleaned it |
| **New/unstained** | **warm, saturated chroma** primary · σ = weak secondary | 0.22–0.30 | 0.70–0.80 | preserve the already-good real grain; light sheen only |

(These replace the client's 0.85/0.45 split — same *idea* of routing, but re-centred
low because we diffuse over a clean base, not raw pixels.)

**Classifier method — REQUIRED stage, SERVER-SIDE, accuracy-first, pre-trained (NO training).**
Decision (client, 2026-07-07): **this is NOT the fence-gate.** The fence-gate is a *tiny* model that
must run **in the browser** on every upload (size + latency constrained). The condition classifier
runs **server-side inside the render pipeline**, where latency is already dominated by the ~2–4 s
diffusion step — so we spend the budget on **accuracy** and use a **large, best-in-class pre-trained
zero-shot** image–text classifier, not a small one.

- **Model — the largest accurate zero-shot classifier that fits the VRAM/latency budget:** e.g.
  **SigLIP 2** (`so400m` or `giant`), or a big OpenCLIP (`ViT-H/14`, `ViT-bigG/14`) / **EVA-CLIP**.
  SigLIP preferred (better-calibrated sigmoid zero-shot). **No "speed" variant** — accuracy first; the
  image encode (~50–200 ms) is negligible beside diffusion. Pick the exact checkpoint by scoring
  candidates on the Phase-0 30-image set. VRAM ~2–5 GB depending on size; runs in the same GPU
  container (or a lighter pre-pass worker if VRAM is tight at HD).
- **Input:** the **masked fence crop** (crop to mask bbox, grey/zero the non-fence pixels) so the
  classifier sees fence, not trees/sky.
- **Label prompts** (candidate set, tuned on the Phase-0 30 images), e.g. *"a weathered grey unstained
  wood fence"*, *"an old fence with green algae and mildew"*, *"bare new unstained wood fence"*, *"a
  freshly stained rich brown wood fence"* → softmax → map the winning cluster to the **weathered** vs
  **clean** profile.
- **Prefer a CONTINUOUS "weathered-ness" score** (softmax prob) over a hard 2-way switch, so denoise /
  ControlNet scale can be **interpolated** between the two profiles (§4.3 table) instead of snapping.
- **Cost/latency:** image encode ~50–150 ms on L4; the fixed label **text embeddings are precomputed
  once at startup**, so per-request cost is just the image encode — negligible vs the diffusion step.
- **Fail-safe fallback:** if SigLIP/CLIP is unavailable or errors, fall back to the cheap **chroma
  heuristic** (weathered = desaturated near-grey; fresh = warm/saturated; green-excess = algae). ⚠ Note
  for that fallback: lead with **chroma**, NOT luminance σ — the proposal's σ signal misroutes because
  heavily weathered wood often has *high* σ (peel/algae/streaks). No bespoke training in any path.

### 4.4 Diffusion implementation detail (verified 2026-07-07, independent review)

Three concrete techniques were proposed (masked control map, a unified inpaint+ControlNet
pipeline, SDEdit "partial noising"). Independent review found all three **technically sound and
standard** — folded in below, re-scoped to our fidelity-first decisions. Two parameters from the
original proposal (**strength 0.65–0.75** and a **color name in the prompt**) are calibrated for a
*different* architecture (prompt-driven SDEdit over the raw original) and are **rejected** — they
break the exact-swatch (ΔE ≤ 3) and same-fence guarantees (see §16).

**(a) DINOv3-masked control map.** Extract Canny/HED edges from the ORIGINAL, then **zero the
background with the DINOv3 mask** so only the fence's structural lines reach ControlNet. This is
additive *beyond* the inpaint mask: the inpaint mask governs *which latents regenerate*, but
ControlNet residuals are computed over the whole frame and can bias generation near the fence
boundary — masking the control map removes that. **Dilate/feather** the mask slightly so perimeter
plank edges aren't clipped.
  - **Depth caveat:** hard-zero is correct for Canny/HED (0 = no edge) but **wrong for a depth map**
    (0 = a real near-plane → a fake depth cliff at the boundary). For the depth signal,
    feather/neutralize/mask-gate the background instead of zeroing.

**(b) Unified pipeline + pinned inputs.** Use the single joint class
`StableDiffusionXLControlNetInpaintPipeline` — inpaint + ControlNet in **one UNet forward per step**
(ControlNet down/mid residuals added each step; masked-latent re-blending handles inpaint; **not**
sequential). Pin inputs unambiguously:
  - `image` / init latents = **the deterministic exact-color BASE render** — NOT the raw original
    (the base already carries the original's shadows + exact swatch + plank identity).
  - `mask_image` = DINOv3 mask (white = fence/regenerate, black = protect background).
  - `control_image` = the DINOv3-masked Canny (3-channel, 0–1, matched resolution).
  - Use the **fp16-fix VAE** (or run VAE in fp32) to avoid SDXL fp16 VAE overflow.
  - Only the masked region is truly generated; unmasked latents are re-blended from the noised base
    each step — this is *why* the background stays pixel-safe regardless of strength.

**(c) SDEdit / partial-noising — what it actually does.** `strength` = the fractional START timestep
of reverse sampling: the init image is noised to t ≈ strength·T, then denoised to 0
(x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε). Because our init image is the deterministic base that ALREADY holds the
original's shadows + exact color, **low strength preserves those low-frequency shadows and plank
identity while diffusion adds high-frequency grain/sheen.**
  - **Corrections to the proposal's framing:** (i) low-freq shadows survive, but *sensor grain is
    high-freq and is destroyed first* — it does NOT "survive"; that's fine, diffusion **adds** fresh
    grain (the whole point). (ii) "strength" and "stop at t≈0.7T" are the **same single knob**, not
    two mechanisms. (iii) 0.65–0.75 is *lower than txt2img (1.0)* but ~2–3× **higher** than our
    0.22–0.40 — a heavy regenerate, not a gentle polish.

**(d) `strength` is a Phase-0 SWEEP, not a constant.** Given D1-A pins the deterministic base, sweep
the **low band ~0.22–0.45** and pick the smallest value that adds convincing grain/sheen while passing
the ΔE / identity / containment gate. Record explicitly: **0.60–0.75 applies ONLY to the rejected
raw-original SDEdit regime** — the two must never be mixed.

**(e) Prompt stays color-neutral.** Texture activators only (e.g. "photorealistic wood grain, premium
wood sealer sheen, outdoor ambient light"). **Reject any color word** ("reddish-brown cedar", "warm"):
a prompt word can't hit an exact Lab swatch to ΔE ≤ 3 and would fight non-brown swatches
(grey/slate/driftwood/ebony). Color is set by the base + D2 color-lock, never the prompt.

### 4.5 Prompt strategy & automated post-processing (verified 2026-07-07)

The tool is **headless** — no user prompt engineering; the backend injects fixed prompts + params on
"Apply Stain." Verified additions:

**Positive prompt — color-neutral AND family-aware.** Texture activators only, reflecting the
customer's chosen stain FAMILY (not a hardcoded one):
> `"freshly applied {family} wood stain, premium wood sealer sheen, visible natural wood grain,
> outdoor sunlight, photorealistic, high resolution"` — `{family}` ∈ {semi-transparent | semi-solid |
> solid}.

No color words (color = base + color-lock). *Rejects the proposal's "rich warm reddish-brown cedar"
and its hardcoded "semi-transparent" regardless of the user's actual family.*

**Negative prompt — new, adopted.** Steers away from our exact failure modes and reinforces the
identity guarantee:
> `"opaque paint, solid flat color, plastic texture, vinyl fencing, high gloss, altered geometry,
> missing or extra planks, warped perspective, hallucinated objects, low quality, blurry"`

**CFG / `guidance_scale` — Phase-0 item, NOT hardcoded 5–6.5.** ⚠ **Interaction:** SDXL-Lightning
(D5) is distilled to run at **low CFG (~1)**, where a negative prompt has little/no effect; CFG 5–6.5
is for *standard* SDXL and will over-burn Lightning. Phase-0 must resolve this as a set (model × steps
× CFG): run Lightning at CFG ~1–2 (negative prompt weak), use a CFG-capable distilled variant, or fall
back to standard SDXL if the negative prompt proves necessary for quality.

**Mask dilation (reinforces §4.4a).** Dilate the fence mask ~1–2 px before inpaint so the stain reaches
the true fence edge and leaves no un-stained weathered "halo" around thin branches. Caveat: don't
over-dilate onto foliage — the containment QA gate (≤ 0.5% bleed) guards this.

**Realism anchor (optional post-step).** After color-lock + composite, optionally blend back a few
percent of the ORIGINAL's **high-frequency residual (camera sensor grain)** inside the mask so the
result isn't too clean/sharp. ⚠ Scope to the high-freq residual **only** — do NOT blend back full
original luminance (that reintroduces the weathering we removed). Amount = config knob.

**Reaffirmed rejections (see §16):** color words in the prompt; `denoising_strength` 0.68–0.72 as a
default (the raw-original heavy-recolor regime — our strength stays 0.22–0.45 over the base).

---

## 5. Component & model selection

| Stage | Model / lib | Why | VRAM (fp16) | Notes / alternatives |
|-------|-------------|-----|------------|----------------------|
| Segmentation | DINOv3 ONNX (existing) | already trained & deployed | ~1.5 GB | reuse `/detect`; may need 1024² mask (currently 512²) — see risk R7 |
| Condition classifier **(required, server-side)** | **large zero-shot SigLIP 2 / big OpenCLIP / EVA-CLIP** (pre-trained) | accuracy-first routing; no training; server GPU — **NOT** the tiny browser fence-gate | ~2–5 GB | pick biggest accurate variant that fits budget; text embeds precomputed; chroma heuristic = fail-safe (§4.3) |
| Base recolor | our intrinsic structure-preserving recolor (Python port) | exact color + identity, cheap | — | port of the shelved canvas algo's *core* (not the failed synth) |
| Depth | Depth-Anything-V2 (small/base) | fast, strong monocular depth | ~1–2 GB | DPT-Hybrid/MiDaS alt (already a training teacher) |
| Edge | HED or Canny (OpenCV) | plank/rail line control | ~0.3 GB | Canny is free; HED cleaner |
| Diffusion | SDXL-Lightning (4–8 step) UNet | realism + sub-second/step | ~7 GB | LCM-LoRA on Juggernaut/RealVisXL alt; SD1.5+CN cheaper fallback |
| ControlNet | depth-sdxl (+ optional tile/canny) | geometry lock | ~2.5 GB each | start depth-only; add tile if surface drifts |
| Color-lock + composite | OpenCV / numpy (Lab or OKLab) | deterministic color guarantee | — | reuse ramp/oklab code |
| Auto-QA | numpy (ΔE, edge-IoU, containment) | gate + fallback trigger | — | logs metrics per render for tuning |

**Peak VRAM** (SDXL-Lightning + 1 depth ControlNet + VAE ~12–15 GB, + DINOv3 seg ~1.5 GB + large
SigLIP/CLIP classifier ~2–5 GB): **~16–21 GB → still fits L4 (24 GB)**, tighter with the biggest
classifiers; **A100 (40/80 GB)** removes the constraint and is ~2–3× faster. The classifier runs at
step [2] *before* diffusion, so if VRAM gets tight it can run as a **pre-pass** (encode → offload)
or in a **separate worker**, decoupled from the diffusion VRAM peak.

---

## 6. The pipeline variants (for the Phase-0 bake-off)

The general technique — pre-tint the fence, then low-denoise diffuse over it with a color-neutral
prompt — is sometimes called **"Pre-Tinted Latent Injection."** It is exactly the plan's spine
(D1-A). The variants differ only in **what the pre-tinted base is** and **how much denoise** is
applied. In ALL variants the deterministic **color-lock (D2-A) is mandatory** — pre-tinting alone
never guarantees ΔE ≤ 3 (diffusion still shifts color; and a *multiply/overlay* tint doesn't even
produce the exact hex — it yields `hex × pixel`, a modulated color — so exact color must be enforced
by the post-lock, not assumed from the tint).

### 6.1 Variant A — Recommended (structure-preserving base + low denoise + color-lock)
As in §4. Base = our intrinsic recolor that **removes weathering** and sets exact color; denoise
0.22–0.40. Highest fidelity, lowest risk. **Default.**

### 6.2 Variant A-flat — Cheap base A/B (flat OpenCV tint + slightly higher denoise)
Base = a flat OpenCV **multiply/overlay** hex tint over the raw original (no weathering removal);
denoise ~0.40–0.50 so diffusion also has to clean the weathering. **Cheaper to build** (no intrinsic
recolor) — worth A/B-ing against A to see if the elaborate base earns its keep. **Risk:** the flat
base still carries the weathering (grey boards/peel/algae) in its luminance, so higher denoise is
needed to hide it, which raises color/identity drift — the exact failure mode the canvas approach hit.
Keep the color-lock.

### 6.3 Variant A′ — Texture-transfer only (most conservative)
Diffusion denoise ≤ 0.20, essentially a "realism polish" over the structure-preserving base.
Nearly zero identity/color risk; realism gain may be modest. Cheap insurance path.

### 6.4 Variant B — Aggressive original (high-denoise, raw-original SDEdit) — see §16
denoise 0.68–0.85 over the raw original. Highest realism ceiling, highest color/identity risk;
**re-opens D1-A** and is out of scope unless escalated (§16). Kept only as a bake-off ceiling reference.

Phase 0 renders all variants on the 30-image test set and scores them on §3 — the data picks the
winner, not opinion. Key question A vs A-flat answers: **does removing weathering in the base (vs
letting diffusion do it at higher denoise) win on ΔE + identity?**

---

## 7. API & integration

New endpoint on the inference service (Cloud Run/Modal), alongside `/detect`:

```
POST /render
  multipart: image (≤20MB), colorHex, family (general|semi-transparent|semi-solid),
             quality (preview|hd), seed (optional), variant (optional, for A/B eval)
  → 200 image/jpeg (HD composited result)
       headers: X-DeltaE, X-Identity, X-Fallback (bool), X-Latency-ms, X-Cache (hit|miss)
  → 422 if not a fence (reuse existing fence-gate logic server-side)
```

**Browser UX (D3-B — every color change renders):**
1. On color pick → a `/render` GPU call produces the full photorealistic result
   (no separate preview tier). A progress spinner covers the ~3–5s warm latency.
2. **Caching is mandatory (D7):** results are keyed by (imageHash, colorHex); re-picking
   any previously-rendered color is instant and free. Optionally pre-warm the cache for
   the most-popular swatches on that image.
3. **Debounce** rapid slider/swatch changes so only the *settled* selection renders —
   never fire a GPU call per intermediate value.
4. Keep the page-load warm-up ping (`GET /`) so the first render of a session starts
   against a warming instance.

> ⚠ **D3-B + D4-B implication:** the *first* color change after an idle period pays a
> cold start (up to §8's cold budget); subsequent changes in the same session are warm
> (back-to-back renders keep the instance alive). Every *uncached* color change is a GPU
> render, so **caching (D7) + debounce are what keep this affordable and responsive.**
> If latency or cost prove painful in Phase 1, the D3-A "instant preview + one-click HD"
> split remains the escape hatch (the deterministic base render already exists as the
> preview).

---

## 8. Latency budget (warm L4, 1024² working res, depth-only ControlNet)

| Stage | Est. time | Notes |
|-------|-----------|-------|
| Segmentation | 0.4–0.9 s | existing; 1024² may add time (R7) |
| Condition + base recolor | 0.2–0.4 s | numpy/OpenCV |
| Depth + edge | 0.3–0.6 s | Depth-Anything-V2 |
| Diffusion (6 steps, 1 CN) | 1.5–3.0 s | SDXL-Lightning; A100 ≈ 0.6–1.2 s |
| Color-lock + composite | ≤ 0.3 s | numpy |
| **Total (warm)** | **~3–5 s** | meets §3 p50 ≤ 4s on a good day; A100 comfortably |
| **Cold start** | **+15–40 s** | SDXL weights load; mitigate with warm pool / bake into image |

If p95 must be < 4s → A100 or SD1.5+ControlNet (D5) or fewer steps.

---

## 9. Cost model (rough, validate in Phase 0)

Two regimes; **the warm instance, not the per-inference compute, is the cost driver.**

- **Scale-to-zero (pilot, D4-B):** pay only while rendering. At ~5 GPU-s/render and
  L4 ≈ $0.6–0.9/GPU-hr equivalent on Cloud Run, marginal cost ≈ **$0.001–0.002 /
  render** + cold-start tax on the first request of an idle period. Cheapest if
  traffic is bursty; UX cost = cold starts.
- **Warm `min-instances=1` (launch, D4-A):** ~**$450–700 / month** for a 24/7 L4,
  regardless of volume. Break-even vs scale-to-zero at roughly a few-hundred
  renders/day. Choose based on real traffic.
- **Caching (D7)** removes repeat-color renders entirely — expected 40–70% hit rate
  once a user explores swatches, so effective cost per *unique* preview is what
  matters.

**D3-B render volume.** Because every color change renders (D3-B), render count ≈
number of *uncached* color picks. The cache (D7) + client debounce are therefore the
dominant cost levers — not the raw per-inference price. Within an active session the
scale-to-zero instance stays warm across back-to-back picks, so only the first pick of
a session pays cold start.

**Recommendation:** pilot scale-to-zero + aggressive cache + debounce; add a warm pool
(`min-instances=1`) at launch only if daily volume or cold-start UX justifies it.
Revisit with real numbers after Phase 1.

---

## 10. Risks & mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| R1 | Output color ≠ selected swatch | High (naive) → Low (with lock) | Critical | Deterministic color-lock §4.2[6] + ΔE auto-QA gate + fallback |
| R2 | "That's not my fence" (identity drift) | High (high-denoise) → Low | Critical | Low denoise, depth+edge ControlNet, diffuse over exact base, identity gate |
| R3 | Latency/cold-start too slow | Medium | High | Warm pool option, LCM/Lightning, A100 tier, "HD on demand" UX |
| R4 | GPU cost per render | Medium | High | Cache by color, scale-to-zero pilot, preview-vs-HD split |
| R5 | Hallucinated artifacts (extra posts, warped planks) | Medium | High | ControlNet + low denoise + auto-QA + fallback to deterministic |
| R6 | Mask bleed onto foliage/sky | Medium | Medium | Hard mask composite §4.2[7] + containment gate |
| R7 | 512² mask too coarse for HD render edges | Medium | Medium | Upscale mask (guided filter) or export higher-res mask from the model |
| R8 | Diffusion nondeterminism between sessions | Medium | Medium | Fixed seed per (image,color); cache result |
| R9 | Model licensing (SDXL/ControlNet/RealVis) for commercial use | Medium | Medium | Verify licenses before launch (SDXL: CreativeML OpenRAIL; check derivatives) |
| R10 | VRAM overflow with multiple ControlNets at HD | Low | Medium | depth-only first; tiled diffusion; A100 |

**Fallback (must-build):** every failure path (timeout, QA fail, GPU error, non-fence)
returns the **deterministic base render** — the user always gets a correct-color,
correct-fence result, just without the diffusion polish.

---

## 11. Phased delivery & go/no-go gates

### Phase 0 — Offline proof-of-concept (no infra, ~2–4 days)
- Build the pipeline as an **offline Python script** on a workstation/Colab GPU.
- Run Variants A / A′ / B on the 30-image test set.
- **Sweep `strength`** in the low band (0.22–0.45, per §4.4d) and ControlNet scale per profile;
  pick the smallest strength that adds convincing grain/sheen while passing the ΔE/identity gates.
- Produce a scorecard (§3 metrics) + side-by-side contact sheet.
- **GATE G0:** at least one variant hits ΔE ≤ 3, identity gate, realism ≥ 4.0, on
  ≥ 80% of the test set. If none do → diffusion is not ready; stop or iterate.

### Phase 1 — Pilot service (behind flag, ~1 week after G0)
- Wrap the winning variant as `/render` on Cloud Run L4 (scale-to-zero).
- Wire the "HD render" button in `index4_dinov3.html` behind a feature flag.
- Add auto-QA + deterministic fallback + caching + metrics logging.
- **GATE G1:** p95 latency, cost/render, and fallback rate within §3 targets on
  real traffic (internal + a few friendly users).

### Phase 2 — Production (~1 week after G1)
- Warm-pool decision (D4), autoscaling limits, monitoring/alerting dashboards.
- WordPress (`wordpress/app.js`) parity + rollout on ninjafencestaining.com.
- Documented runbook (cold-start, OOM, cost alarms).
- **GATE G2:** stable for N days, cost within budget, no color/identity complaints.

Each gate is a real stop/continue — we don't proceed on hope.

---

## 12. Evaluation harness (built in Phase 0, reused forever)

- `eval/render_eval.py`: runs a variant over the test set, writes per-image
  {ΔE, edge-IoU, containment, latency} + a montage HTML.
- Ground-truth swatches: the exact hex/Lab of each catalog stain.
- Human-rating sheet (realism, "same fence?") for the subjective gates.
- This harness is the arbiter for D1/D5/D6 and every future model bump.

---

## 13. What we reuse vs build

**Reuse:** DINOv3 segmentation + Cloud Run/Modal serving skeleton; the *core*
intrinsic recolor math (color-lock, OKLab ramp) from the shelved canvas work; the
fence-gate; the page-load warm-up ping; the upload/CORS plumbing.

**Build new:** `/render` endpoint; diffusion + ControlNet + depth stages; condition
classifier; auto-QA + fallback; caching layer; eval harness; HD-render UX + flag.

---

## 14. Rough effort estimate (engineering)

| Phase | Effort | GPU cost |
|-------|--------|----------|
| Phase 0 POC + eval | 2–4 days | a few $ of on-demand GPU |
| Phase 1 pilot service | ~1 week | pilot inference + minimal warm |
| Phase 2 production | ~1 week | ongoing per §9 |

---

## 15. Decision log (fill in as we go)
- 2026-07-07 — Plan drafted; recommended Variant A (low-denoise + color-lock).
- 2026-07-07 — **Client locked D1–D7:** D1=A (low-denoise), D2=A (color-lock),
  **D3=B (every color change renders)**, D4=B (scale-to-zero), D5=SDXL-Lightning 4–8 step,
  D6=Cloud Run L4, D7=Yes (cache by imageHash+colorHex). **#1 hard requirement confirmed:
  output must match the exact commercial swatch color** → color-lock mandatory, ΔE is the
  primary acceptance gate. D3-B adopted → caching + debounce are the cost/latency levers
  (§7, §9). **Not executing yet** — awaiting "execute Phase 0".
- 2026-07-07 — Independent review of the client's 3 proposed diffusion techniques (masked control
  map, unified `StableDiffusionXLControlNetInpaintPipeline`, SDEdit partial-noising): all **sound and
  adopted** (§4.4). **Rejected** the proposal's `strength` 0.65–0.75 default and its color-named prompt
  as a different (raw-original) architecture that breaks ΔE ≤ 3 + same-fence → recorded as change
  control (§16). Corrected two framing errors: (a) sensor grain is high-freq and is *regenerated*, not
  preserved; (b) `strength` and "stop at t≈0.7T" are one knob, and 0.7 is ~2–3× *higher* than our
  low-denoise, not "gentle."
- 2026-07-07 — Reviewed a 3rd proposal (headless hardcoded prompts + "golden" hyperparams + edge-case
  post-processing). **Adopted (§4.5):** a **negative prompt**; a color-neutral, **family-aware** positive
  prompt; the mask-dilation "no weathered halo" rationale; and an optional **high-freq realism-anchor**
  blend. **Flagged** the CFG × negative-prompt × SDXL-Lightning interaction as a Phase-0 item (don't
  hardcode CFG 5–6.5 — wrong for Lightning; negative prompt is weak at CFG≈1). **Re-rejected** (again)
  color words in the prompt and `denoising_strength` 0.68–0.72 (§16). Scoped the realism-anchor to the
  high-freq residual only (full-luminance blend would reintroduce weathering).
- 2026-07-07 — Reviewed "Pre-Tinted Latent Injection": it IS the plan's D1-A spine (pre-tint +
  low-denoise + color-neutral prompt) — nothing fundamentally new. **Corrections:** a multiply/overlay
  tint ≠ exact hex (it yields `hex × pixel`), and pre-tint alone never guarantees ΔE — the **color-lock
  (D2-A) is the guarantee**, and the proposal omitting it is a gap. **Added Variant A-flat (§6.2)** — its
  flat-tint base as a cheaper Phase-0 A/B vs the structure-preserving base; flagged that its 0.40–0.50
  denoise over a weathering-carrying base is the higher-risk regime (the canvas failure mode).
- 2026-07-07 — **Condition classifier decided: REQUIRED stage, primary method = pre-trained zero-shot
  SigLIP/CLIP** on the masked fence crop (client choice) — no training. Chroma heuristic demoted to a
  fail-safe fallback; luminance-σ corrected to a weak secondary (misroutes on peel/algae). Prefer a
  continuous weathered-ness score to interpolate denoise between profiles. VRAM +1–2 GB, still fits L4.
  (§4.1[2], §4.3, §5.)
- 2026-07-07 — **Classifier sizing clarified (client):** it is **server-side** and should be a **large,
  best-accuracy** model (SigLIP 2 `so400m`/`giant`, big OpenCLIP `ViT-H`/`bigG`, or EVA-CLIP) —
  explicitly **NOT** the tiny browser fence-gate model. Accuracy-first (encode ~50–200 ms is negligible
  vs diffusion). VRAM revised to ~2–5 GB, still fits L4; runs as a pre-pass/worker if tight.

---

## 16. Change control — rejected as implementation detail (escalate if the client insists)

The originally-proposed **`strength` 0.65–0.75 over the raw original with a color-named prompt** is
**not an implementation detail** — it is a *different architecture* (prompt-driven SDEdit recolor)
that re-opens locked **D1-A** and loosens fidelity. Independent review (2026-07-07) found it cannot
meet ΔE ≤ 3 or the same-fence guarantee: at strength ≈ 0.7 most of x₀ is drowned, so the exact swatch
and plank identity the base established are largely **regenerated**, overloading the color-lock (which
fixes a*/b* but inherits an invented L* → flat/washed darks) and tripping the identity/containment QA
into fallback anyway — wasting the GPU pass. If the client wants this regime, **escalate it as a formal
scope change**, do not merge it silently. The plan proceeds with low-strength diffusion over the
deterministic exact-color base.
