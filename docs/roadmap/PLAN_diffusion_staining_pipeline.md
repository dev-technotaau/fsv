# Plan — Fence Restaining (lean, self-hosted Qwen)

**Status:** active · **Date:** 2026-07-08
**Supersedes:** the SDXL-Inpaint + ControlNet + condition-classifier plan (over-engineered, wrong model class — see §"What we deleted").

---

## 1. The one insight that changes everything

**Tinting ≠ renovating.** The current JS (and the old plan) keep the weathered wood's
**luminance** and swap the colour → the grey streaks, water stains, peeling and algae all
show through. That's why `fence_stained.jpg` still looks old under a tan wash.

Gemini didn't tint — it **regenerated the wood surface**: fresh clean planks, new grain,
uniform tone. To match that we need an **instruction-conditioned generative editor**, not
SDXL + ControlNet + denoise tuning. That whole family can only *tint safely* — it can't
*renew* — which is why every variant we tried came back burnt-out or still-weathered.

## 2. Decision (2026-07-08, client)

- **Fully self-hosted, no external API** (Gemini API rejected — must own it).
- **Engine = Qwen-Image-Edit-2509** (Alibaba, **Apache-2.0**, commercial-OK). Genuine
  renovation, ~85–95% of the Gemini bar.
- **Accept ~30 s async** on the existing Cloud Run **24 GB L4**. Interactive ≤15 s is NOT
  possible for a renovator on an L4 (that needs an A100, or paid-license FLUX + an
  always-warm instance — both rejected). So the UX is submit → ~30 s → HD result.

## 3. The pipeline (3 stages — that's it)

```
1. SEGMENT   DINOv3 mask (reuse the existing /detect model)
2. RENOVATE  Qwen-Image-Edit-2509: "restain this fence to fresh, clean, new {family}
             wood — clean planks, natural grain, remove all weathering/algae/peeling;
             keep the plank layout, lighting and everything else identical."
3. FINISH    (CPU, model-agnostic)
             a. color-lock the fence to the EXACT swatch hex  → ΔE ≤ 3 guaranteed
             b. composite fence-only over the ORIGINAL via the mask → background pristine
```

Qwen re-renders the whole frame and only needs to *renovate convincingly* — it does NOT
need to hit the exact hex (the color-lock owns that) or keep the background (the composite
owns that). That division is what makes the whole thing robust.

## 4. What we deleted (the over-engineering)

variant ladder (A/A-flat/A′/B/RM/…), the condition classifier (SigLIP/CLIP weathered-vs-clean
routing), multi-ControlNet + masked control maps + depth caveats, the structure-preserving
intrinsic base recolor, denoise sweeps, the QA-gate/eval-harness infra, and the D1–D7
cost/caching/warm-pool decision matrix. **None of it produces renovation** — it was all
scaffolding around a model class that can only tint.

## 5. What we kept (good bones, model-agnostic)

- **Color-lock** — re-impose the exact swatch chroma as a post-step (ΔE ≤ 3).
- **Mask composite** — fence from the edit, background pixel-identical from the original.
- **A ΔE sanity check.**

## 6. Self-host config (to fit 24 GB + reach ~30 s)

- Qwen bf16 ≈ 40 GB → **won't fit** 24 GB. Quantize: **Nunchaku INT4 SVDQuant** (~8–12 GB)
  or GGUF-Q4 (~18 GB), plus the official **Lightning 4/8-step LoRA**.
- Latency on the L4: **~30–45 s @1024px**, ~18–25 s @768px (softer grain). The 8 B
  Qwen2.5-VL encoder adds ~4–6 s fixed overhead that step-cutting can't remove.
- Cold start reloads the weights (+8–60 s). Accept it for testing, or run one warm
  instance later (only real recurring cost; decide at launch).

## 7. Milestones

- **M0 — validate quality (now).** Run Qwen-Image-Edit-2509 at its *best* quality (speed
  irrelevant) on real weathered-fence photos. **GATE: does it renovate like Gemini?**
  If no → stop and reconsider. If yes → build the rest.
- **M1 — /render on Cloud Run.** Qwen INT4 + Lightning, async job, + the CPU finisher.
- **M2 — wire in** `index4_dinov3.html` + `wordpress/app.js` behind a flag; submit→poll UX.

## 8. Cost

Per-image GPU-seconds are trivial. The only recurring cost is an optional warm L4
(~$480/mo) to avoid cold starts — deferred to launch. Testing ≈ free.
