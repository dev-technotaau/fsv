# AREA: Web frontend assets, browser/server models, and deployment (HuggingFace Spaces + WordPress + Cloudflare)

## SUMMARY
The fence-staining-visualizer directory holds the entire web deliverable: six index*.html iterations, three legacy browser-side ONNX models, a WebAssembly post-processing module, and a WordPress plugin package. The CURRENT production frontend is index4_dinov3.html (and its WordPress twin wordpress/app.js + fence-simulator-body.html), which does NOT run any model in the browser — it uploads a downscaled JPEG to a server-side DINOv3 ViT-L/16 endpoint hosted on Google Cloud Run (https://fsv-dinov3-467125191853.us-central1.run.app/detect) and receives a 512x512 PNG mask. The three in-browser ONNX models (UNet++, SegFormer-B3, DINOv2-Small) plus best_unetpp_v2.pth are all LEGACY/experimental artifacts from earlier client-side-inference iterations that the server-side DINOv3 architecture replaced. There are two parallel production deployments of the same code: a standalone static HTML build (GitHub Pages / HuggingFace Spaces static SDK) and a Shadow-DOM WordPress plugin embedded at ninjafencestaining.com via the [fence_simulator] shortcode. The README.md is badly out of date — it still describes the original UNet++ client-side product.

## KEY_FACTS
- The CURRENT production page is index4_dinov3.html (542,380 bytes, modified Jun 9 2025), the largest and newest HTML file in the directory.
- index4_dinov3.html runs NO model in the browser: it POSTs a downscaled JPEG to CONFIG.MODAL_ENDPOINT = 'https://fsv-dinov3-467125191853.us-central1.run.app/detect' (index4_dinov3.html:2899) and receives a 512x512 PNG mask. Despite the constant name 'MODAL_ENDPOINT', the URL is a Google Cloud Run host (us-central1.run.app), not modal.run.
- The server-side model is DINOv3 ViT-L/16; the page comment says INPUT_SIZE=512 because DINOv3 patch_size=16 (32*16=512 exactly) and explicitly warns NOT to use 518 which is DINOv2's 37*14 value (index4_dinov3.html:2923-2929).
- Browser upload config: UPLOAD_MAX_DIM=1024, UPLOAD_JPEG_QUALITY=0.85, POSTPROCESS_MAX_DIM=1536, CLEAN_MAX_DIM=1536, DEFAULT_THRESHOLD=0.5 (index4_dinov3.html:2900-2930).
- DINOv3-L post-processing tuning: CC_MIN_BLOB_AREA_PCT=0.05 (lowered from 2.0 which was DINOv2-S-tuned), CC_KEEP_TOP_K_BLOBS=0, POST_ERODE_PX=0, USE_SOFT_MASK=true with SOFT_MASK_LOW lowered 0.70->0.55 to fix the dark-fence-gap failure mode (index4_dinov3.html:2939-2972).
- fence_model_dinov2.onnx (135,612,248 bytes / 135.6 MB) is FenceSegmentationModel: backbone=facebook/dinov2-small, decoder=mask2former dim=192 q=16 L=6, refinement_iters=2; input 1x3x518x518, patch_size=14, val_iou=0.4253 at epoch 19, opset 18 (fence_model_dinov2.json).
- fence_model_segformer.onnx (178,859,708 bytes / 178.9 MB) is smp.Segformer(encoder_name='mit_b3', classes=1), trained with FocalTversky loss, EMA weights, best_iou=0.4550 at epoch 18, input 1x3x512x512, opset 17 (fence_model_segformer.json).
- fence_model_unet_browser.onnx (35,200,045 bytes / 35.2 MB, dated Nov 11 2025) is the original UNet++ browser model the README describes; it is the oldest model artifact.
- best_unetpp_v2.pth (714,499,181 bytes / 714.5 MB) is the SegFormer-B3 source checkpoint despite the 'unetpp' filename — convert_to_onnx.py loads it as smp.Segformer mit_b3 and notes '~715 MB raw .pth' (convert_to_onnx.py:3, 35, 77-82).
- convert_to_onnx.py converts best_unetpp_v2.pth -> fence_model_segformer.onnx, bakes sigmoid into the graph, reads EMA weights, uses legacy torch.onnx.export(dynamo=False) for single-file output, ImageNet normalize mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225], opset 17, parity tol 5e-3 (convert_to_onnx.py).
- convert_web_deployable_to_onnx.py converts outputs/web_deployable/web_v1/checkpoints/best_inference.pt -> fence_model_dinov2.onnx; model is DINOv2-Small + ViTToFPN + MSDeform 6L + Mask2Former 6L 192-dim 16-query + UNet3+ refinement; auto-snaps 512->518 for patch_size=14, bakes sigmoid + temperature=1.0 (convert_web_deployable_to_onnx.py:1-21).
- ONNX parity checks pass: dinov2 max_abs_diff=1.74e-05, segformer max_abs_diff=1.90e-10, both within 5e-3 tolerance (the two .json sidecars).
- HTML lineage: index.html (87KB, SegFormer-B3 browser ONNX, MODEL_PATH='./fence_model_segformer.onnx', InferenceSession), index2.html / index3.html / index4.html (DINOv2-Small server endpoint 'https://dev-45325--f-stain-inference-web.modal.run/detect'), index2_dinov3.html (DINOv3-L Phase1 endpoint 'https://dev-45325--f-stain-dinov3-inference-web.modal.run/detect'), index4_dinov3.html (DINOv3-L on Cloud Run, CURRENT).
- Earlier pages reference real Modal endpoints (dev-45325--*.modal.run); the production index4_dinov3.html and WordPress app.js have migrated to Cloud Run (us-central1.run.app), evidence of a Modal->Cloud Run hosting migration.
- WordPress deployment: fence-stain-simulator.php (Plugin v2.0.0, Author TechnoTaau, License Proprietary, Plugin URI ninjafencestaining.com) registers a [fence_simulator] shortcode that outputs a <fence-simulator> custom element; checks both post_content and Elementor _elementor_data postmeta for the shortcode (fence-stain-simulator.php:6-44, 153-161).
- fsv-loader.js defines the <fence-simulator> custom element, attaches an open Shadow DOM, fetches app.css + fence-simulator-body.html into the shadow root, injects Google Fonts + Bootstrap Icons <link> inside the shadow, then calls window.FSV_initFenceSimulator(shadow) from app.js (fsv-loader.js).
- WordPress app.js (207,586 bytes) is the production logic derived from index4_dinov3.html and uses the SAME Cloud Run endpoint 'https://fsv-dinov3-467125191853.us-central1.run.app/detect' (wordpress/app.js:20-21).
- .htaccess adds 'AddType application/wasm .wasm' (Apache) so the post-process WASM loads via WebAssembly.instantiateStreaming; notes Nginx ignores it (wordpress/.htaccess).
- WASM post-processing: wasm/fsv_postprocess.c (21,710 bytes) compiles to two artifacts via wasm/Makefile (emscripten) — fsv_postprocess.wasm (9,668 bytes, scalar) and fsv_postprocess_simd.wasm (19,725 bytes, SIMD128 ~2-3x faster); JS feature-detects SIMD via WebAssembly.validate. Build outputs in wasm/build/ also include .js glue files.
- scripts/bake_fence_gate_features.mjs offline-precomputes SigLIP (Xenova/siglip-base-patch16-224, quantized) text embeddings for 70 prompts, embeddingDim 768, output fence_gate_text_features_v1.bin (215,040 bytes) + .json, dropping gate-ready time from ~36s to ~50ms (bake script + fence_gate_text_features_v1.json).
- The SigLIP baked text-features are now STALE: the CURRENT in index4_dinov3.html fence pre-filter uses a zero-shot OBJECT DETECTOR, Xenova/owlv2-base-patch16-ensemble (~150 MB quantized, cached in IndexedDB), explicitly chosen OVER SigLIP/CLIP classification; gate is fail-open returning {isFence:true} on any error (index4_dinov3.html:4006-4047).
- .wrangler/cache/wrangler-account.json ties the Cloudflare account to id c6027029a4faaf83fb46084c06d9762b, name 'Stumpgrindinghouston@gmail.com's Account' — evidence of Cloudflare Wrangler (Pages/Workers) deployment.
- fence-staining-visualizer/.gitignore excludes *.pth, *.pt, *.zip, checkpoints/, logs/, data/{images,masks,annotations} — so best_unetpp_v2.pth and the wordpress .zip packages are intentionally untracked locally-present artifacts.
- README.md (Nov 18 2025) describes the OLD product: UNet++ with EfficientNet-B7 encoder, 512x512, 100% client-side browser ONNX, 'no server uploads', trained on 800 images (680 train / 120 val) — directly contradicting the current server-side DINOv3 architecture.

## FILE_ROLES
- [stale] fence-staining-visualizer/README.md — Project README; describes the ORIGINAL UNet++ 100%-client-side product. Badly out of date vs current server-side DINOv3 deployment.
- [current] fence-staining-visualizer/index4_dinov3.html — CURRENT production standalone page; DINOv3 ViT-L/16 via Cloud Run server endpoint, OWLv2 client pre-filter, WASM post-process.
- [stale] fence-staining-visualizer/index2_dinov3.html — Earlier DINOv3-L (Phase1 ep1) page pointing at a Modal endpoint; superseded by index4_dinov3.html.
- [stale] fence-staining-visualizer/index4.html — DINOv2-Small server-endpoint page; predecessor UI to index4_dinov3.html.
- [stale] fence-staining-visualizer/index3.html — DINOv2-Small server-endpoint page (Modal); intermediate experiment.
- [stale] fence-staining-visualizer/index2.html — DINOv2-Small server-endpoint page (Modal); early v2 UI.
- [stale] fence-staining-visualizer/index.html — SegFormer-B3 IN-BROWSER ONNX page (MODEL_PATH=./fence_model_segformer.onnx); the last fully client-side inference page.
- [stale] fence-staining-visualizer/fence_model_dinov2.onnx — Browser ONNX (135.6 MB) DINOv2-Small FenceSegmentationModel; experimental client-side model, not used by current server pipeline.
- [stale] fence-staining-visualizer/fence_model_dinov2.json — Sidecar metadata for the DINOv2 ONNX (input 518, val_iou 0.4253, opset 18).
- [stale] fence-staining-visualizer/fence_model_segformer.onnx — Browser ONNX (178.9 MB) SegFormer-B3; used by index.html legacy client-side page.
- [stale] fence-staining-visualizer/fence_model_segformer.json — Sidecar metadata for the SegFormer ONNX (best_iou 0.4550, opset 17).
- [stale] fence-staining-visualizer/fence_model_unet_browser.onnx — Original UNet++ browser model (35.2 MB) the README documents; oldest model artifact.
- [artifact] fence-staining-visualizer/best_unetpp_v2.pth — 714.5 MB SegFormer-B3 source checkpoint (misnamed 'unetpp'); input to convert_to_onnx.py. Gitignored.
- [stale] fence-staining-visualizer/convert_to_onnx.py — Converts best_unetpp_v2.pth -> fence_model_segformer.onnx (sigmoid baked, EMA, opset 17).
- [stale] fence-staining-visualizer/convert_web_deployable_to_onnx.py — Converts web_deployable best_inference.pt -> fence_model_dinov2.onnx (DINOv2 client model).
- [current] fence-staining-visualizer/wordpress/fence-stain-simulator.php — CURRENT WordPress plugin (v2.0.0) registering [fence_simulator] shortcode + Shadow-DOM asset enqueue.
- [current] fence-staining-visualizer/wordpress/fsv-loader.js — CURRENT custom-element loader; builds Shadow DOM, fetches css/html, hands off to app.js.
- [current] fence-staining-visualizer/wordpress/app.js — CURRENT WordPress simulator logic (207 KB) derived from index4_dinov3.html; uses Cloud Run DINOv3 endpoint.
- [backup] fence-staining-visualizer/wordpress/app.js.bak — Backup of a previous app.js (129 KB).
- [current] fence-staining-visualizer/wordpress/app.css — CURRENT styling injected into the Shadow DOM (56 KB).
- [current] fence-staining-visualizer/wordpress/fence-simulator-body.html — CURRENT body markup fetched into the shadow root (modals, canvases, controls).
- [current] fence-staining-visualizer/wordpress/index.html — Standalone WP test harness loading app.css + app.js directly (no WP).
- [current] fence-staining-visualizer/wordpress/.htaccess — Apache MIME config: AddType application/wasm .wasm.
- [current] fence-staining-visualizer/wordpress/fence-stain-simulator.zip — Latest packaged plugin zip (286 KB, Jun 9); shippable WordPress upload.
- [backup] fence-staining-visualizer/wordpress/fence-stain-simulator-latest-v5.zip — Versioned plugin package (v5); superseded by fence-stain-simulator.zip.
- [backup] fence-staining-visualizer/wordpress/fence-stain-simulator-latest-v4.zip — Versioned plugin package (v4).
- [backup] fence-staining-visualizer/wordpress/fence-stain-simulator-latest-v3.zip — Versioned plugin package (v3).
- [backup] fence-staining-visualizer/wordpress/fence-stain-simulator-latest-v2.zip — Versioned plugin package (v2).
- [backup] fence-staining-visualizer/wordpress/fence-stain-simulator-latest.zip — Versioned plugin package (latest, early).
- [backup] fence-staining-visualizer/wordpress/fence-stain-simulator-copy.zip — Plugin package copy.
- [current] fence-staining-visualizer/wasm/fsv_postprocess.c — C source for the WASM post-process hot loops (Lab conversion, box blur, sliding max/min).
- [current] fence-staining-visualizer/wasm/Makefile — Emscripten build for scalar + SIMD .wasm; deploy target copies into hosting dirs.
- [current] fence-staining-visualizer/wasm/build/ — Built WASM + JS glue (fsv_postprocess.js/.wasm and _simd variants).
- [current] fence-staining-visualizer/fsv_postprocess.wasm — Deployed scalar WASM post-process module (9.7 KB) served alongside the HTML.
- [current] fence-staining-visualizer/fsv_postprocess_simd.wasm — Deployed SIMD128 WASM post-process module (19.7 KB).
- [stale] fence-staining-visualizer/scripts/bake_fence_gate_features.mjs — Offline SigLIP text-embedding baker for the OLD classification-based fence gate.
- [stale] fence-staining-visualizer/scripts/package.json — npm config for the bake script (@xenova/transformers ^2.0.1).
- [stale] fence-staining-visualizer/fence_gate_text_features_v1.json — Baked SigLIP embedding metadata (70 prompts, dim 768); superseded by OWLv2 gate.
- [stale] fence-staining-visualizer/fence_gate_text_features_v1.bin — Baked SigLIP text embeddings (215 KB raw Float32); superseded by OWLv2 gate.
- [artifact] fence-staining-visualizer/.wrangler/cache/wrangler-account.json — Cloudflare Wrangler account cache (account id + name) indicating Pages/Workers deploy.
- [current] fence-staining-visualizer/.gitignore — Ignores *.pth/*.pt/*.zip/checkpoints/logs/data; explains why heavy artifacts are untracked.
- [stale] fence-staining-visualizer/assets/ — README demo images (fence_sample_1.jpg, mask_sample_1.png, screenshot_demo.png).
- [artifact] fence-staining-visualizer/cc_cleanup_demo.png — Connected-component cleanup demo screenshot (dev artifact).
- [artifact] fence-staining-visualizer/dinov2_vs_segformer_smoke.png — Model comparison smoke-test image (dev artifact).
- [artifact] fence-staining-visualizer/dinov2_vs_segformer_smoke_v2.png — Model comparison smoke-test image v2 (dev artifact).
- [artifact] fence-staining-visualizer/onnx_smoke_test_output.png — ONNX smoke-test output image (dev artifact).
- [artifact] fence-staining-visualizer/postprocess_comparison.png — Post-process comparison image (dev artifact).

## NARRATIVE
## Web frontend, browser/server models, and deployment

The `fence-staining-visualizer/` folder is the customer-facing end of the whole project, and the single most important thing to understand about it is that the directory tells the story of a product that migrated, twice, away from running anything in the browser. If you read the README cold you would conclude this is a privacy-friendly, 100%-client-side UNet++ tool that never uploads a photo. That description is now historical fiction. The shipping product uploads a downscaled JPEG to a server-side DINOv3 ViT-L/16 model and gets back a mask. Everything else in the folder is either glue around that, or a fossil from an earlier era.

### The HTML lineage (six files, one winner)

There are six `index*.html` files and they form a clean evolutionary chain. The earliest surviving page, `index.html` (87 KB), is the last fully self-contained client-side build: it sets `MODEL_PATH = './fence_model_segformer.onnx'` and instantiates an ONNX Runtime Web session right in the browser (`ort.InferenceSession.create`). It is a SegFormer-B3 model with sigmoid baked into the graph and ImageNet normalization done in JS.

Then the team moved inference off the device. `index2.html`, `index3.html`, and `index4.html` all switch to a server endpoint — a DINOv2-Small model behind a Modal URL (`https://dev-45325--f-stain-inference-web.modal.run/detect`). These three are progressively more elaborate UIs (index4.html is 346 KB) but architecturally identical: send a JPEG, receive a mask. `index2_dinov3.html` is the first DINOv3 page — it labels itself "Phase 1, ep 1" and points at a separate Modal endpoint (`...f-stain-dinov3-inference-web.modal.run/detect`).

The current production page is **`index4_dinov3.html`** — 542 KB, the newest file (Jun 9), and the only one that combines the flagship DINOv3-L backbone with the mature post-processing pipeline. Crucially, its endpoint is no longer Modal: `CONFIG.MODAL_ENDPOINT = 'https://fsv-dinov3-467125191853.us-central1.run.app/detect'`. The constant kept the historical name "MODAL_ENDPOINT," but the host is Google **Cloud Run** (the numeric `467125191853` is a GCP project number). So there's a second migration baked into the constants: Modal serverless → Cloud Run. The page's own comments are emphatic that this is server-side: "Inference runs server-side on Modal. Browser only uploads photos," and the model loads "13 MB ONNX (+ 2.3 GB external data) once and reuses it across requests." The browser never downloads a segmentation model.

The DINOv3 input handling is worth a line for the report because it's a real correctness fix: `INPUT_SIZE = 512`, with a loud warning never to use 518. 518 is DINOv2's value (37×14 for patch_size 14); DINOv3 uses patch_size 16, so the trained grid is 32×16 = 512, and 518÷16 is not integral, which produced visible horizontal banding when the mask was upsampled. That kind of comment is the signature of a bug that actually shipped once.

### The "browser models" are legacy

Three ONNX files and one `.pth` sit in the root and look like production assets but aren't, for the current pipeline:

- `fence_model_unet_browser.onnx` (35.2 MB) — the original UNet++ model the README is written around. Oldest artifact (Nov 11).
- `fence_model_segformer.onnx` (178.9 MB) — SegFormer-B3 (mit_b3), trained with FocalTversky loss, EMA weights, best IoU 0.4550 at epoch 18. Still referenced by the legacy `index.html`.
- `fence_model_dinov2.onnx` (135.6 MB) — the DINOv2-Small `FenceSegmentationModel` (DINOv2-S backbone + Mask2Former decoder dim 192 / 16 queries / 6 layers + 2-iter refinement), val IoU 0.4253 at epoch 19.
- `best_unetpp_v2.pth` (714.5 MB) — confusingly named, but it is actually the SegFormer-B3 source checkpoint; `convert_to_onnx.py` opens it and constructs `smp.Segformer(encoder_name="mit_b3")`. It's gitignored (`*.pth`), so it's a local artifact only.

Both conversion scripts (`convert_to_onnx.py` → segformer, `convert_web_deployable_to_onnx.py` → dinov2) follow the same recipe: load EMA weights, wrap the model so sigmoid is baked into the graph, export with the legacy `torch.onnx.export(dynamo=False)` so weights stay in a single file (the new dynamo exporter spills weights to an external `.onnx.data`, which is bad for a single-file browser download), then run an onnxruntime parity check against PyTorch. Parity is tight: max abs diff 1.9e-10 (segformer) and 1.7e-05 (dinov2), both well under the 5e-3 tolerance. The sidecar JSONs document exactly how the browser should preprocess (divide by 255, ImageNet mean/std, NCHW) and threshold (0.5). All of this is real, working engineering — it's just no longer on the production path, because the DINOv3-L model is too large to ship to a browser and lives on Cloud Run instead. For the client report these should be framed as the R&D ladder (UNet++ → SegFormer → DINOv2 client → DINOv3 server) rather than current deliverables.

### The client-side fence pre-filter (and a stale sub-pipeline)

Before any photo is sent to the server, `index4_dinov3.html` runs a client-side zero-shot gate to reject obvious non-fence uploads (selfies, food, a coffee cup on a wooden table). The CURRENT gate is a zero-shot **object detector**, `Xenova/owlv2-base-patch16-ensemble` (~150 MB quantized, cached in IndexedDB), chosen deliberately over CLIP/SigLIP classification because detection can localize a small fence in a busy scene where whole-image classification fails. The gate is fail-open: any error returns `{isFence: true}` and lets the server decide.

This is where another fossil lives. `scripts/bake_fence_gate_features.mjs` plus `fence_gate_text_features_v1.{bin,json}` are an offline-baked set of **SigLIP** text embeddings (`Xenova/siglip-base-patch16-224`, 70 prompts, 768-dim, ~215 KB) built to make a *classification*-based gate start in ~50 ms instead of ~36 s. Since the production gate moved to OWLv2 detection, those baked features and the bake script are stale. They're still copied into both the standalone dir and the WordPress dir, which is why you see the `.bin`/`.json` in two places.

### WebAssembly post-processing

The mask coming back from the server still needs heavy client-side cleanup (Lab color conversion, masked box blur, sliding-window morphology, connected-component filtering). The hot loops are implemented in C (`wasm/fsv_postprocess.c`, 21.7 KB) and compiled by `wasm/Makefile` via Emscripten into two artifacts: a portable scalar `fsv_postprocess.wasm` (9.7 KB) and a SIMD128 build `fsv_postprocess_simd.wasm` (19.7 KB, ~2–3× faster). The JS loader feature-detects SIMD with `WebAssembly.validate()` and picks the right one. These `.wasm` files are deployed both at the root and inside the WordPress plugin, and there's an `.htaccess` whose only job is `AddType application/wasm .wasm` so Apache serves the correct MIME type and the browser can use `instantiateStreaming`.

### Two production deployments of one codebase

There are two live targets, both running the same DINOv3 Cloud Run backend:

1. **Standalone static HTML** — `index4_dinov3.html` served as a static site. The README badges point at GitHub Pages (`technotaau.github.io/fence-staining-visualizer`), and per the project memory the HuggingFace Spaces static SDK serves it from a `.static.hf.space` subdomain. No server component on the hosting side; the only backend is Cloud Run. The presence of `.wrangler/cache/wrangler-account.json` (Cloudflare account `c6027029a4faaf83fb46084c06d9762b`) shows a Cloudflare Wrangler (Pages/Workers) deploy path is also wired up.

2. **WordPress plugin** — the `wordpress/` folder is a self-contained plugin (`fence-stain-simulator.php`, "Plugin Name: Fence Stain Simulator," v2.0.0, Author TechnoTaau, License Proprietary, URI ninjafencestaining.com). It registers a `[fence_simulator]` shortcode that emits a single `<fence-simulator>` custom element. The clever part is the Shadow DOM isolation: `fsv-loader.js` defines that custom element, attaches an open shadow root, fetches `app.css` and `fence-simulator-body.html` into it, and injects Google Fonts + Bootstrap Icons `<link>`s *inside* the shadow (the PHP deliberately does NOT register `app.css` at document level, so the host theme's CSS can never reach in and the simulator's CSS can never leak out). `app.js` (207 KB — the WordPress port of `index4_dinov3.html`'s logic) defines `window.FSV_initFenceSimulator(shadow)`, which the loader calls once the shadow tree is laid out. The PHP also handles the real-world messiness of page builders: it checks Elementor's `_elementor_data` postmeta for the shortcode (because `has_shortcode()` against `post_content` returns false on Elementor pages) and re-enqueues assets from the shortcode handler as a late safety net. `app.js` uses the identical Cloud Run endpoint as the standalone page.

The `wordpress/` folder also carries a lot of packaging history: `app.js.bak`, and seven `fence-stain-simulator*.zip` files (copy, latest, latest-v2..v5, and the newest plain `fence-stain-simulator.zip` at 286 KB). The plain `.zip` is the current shippable plugin; the versioned ones are backups. All zips are gitignored (`*.zip`).

### Bottom line for the report

The honest narrative is: the client product is a thin browser UI (standalone HTML and a Shadow-DOM WordPress embed) that does smart client-side gating (OWLv2) and WASM post-processing, but delegates the actual fence segmentation to a server-hosted DINOv3 ViT-L/16 model on Google Cloud Run. The various browser ONNX models, the `best_unetpp_v2.pth`, the conversion scripts, the SigLIP feature baker, and the README's "100% client-side UNet++" claim are all earlier chapters, not the current product.

## UNCERTAINTIES
- The exact HuggingFace Spaces URL is not present in any file I read; the .static.hf.space subdomain pattern comes from project memory, not a citation in this directory. The README badges point at GitHub Pages (technotaau.github.io / chanderbhanswami.github.io), not HF Spaces.
- The Cloud Run endpoint host (fsv-dinov3-467125191853.us-central1.run.app) is GCP Cloud Run, but the constant is still named MODAL_ENDPOINT and comments still say 'Modal.' I cannot confirm whether the live backend is actually Cloud Run or Modal without checking the cloudrun_inference/ and modal_inference/ folders (outside this subsystem). The naming is contradictory.
- The Cloudflare Wrangler account is registered to 'Stumpgrindinghouston@gmail.com' — unclear whether this is the client's account or a contractor's; no wrangler.toml was found in this directory to confirm what is actually deployed via Cloudflare (Pages vs Workers vs nothing).
- I did not exhaustively read all 542 KB of index4_dinov3.html or all 207 KB of wordpress/app.js line-by-line; my model/endpoint/config claims are from targeted reads of the CONFIG block, the pre-filter section, and grep over model/endpoint keywords. There could be additional fallback endpoints I did not surface (the grep showed a 'cascading fallback' comment at line 5042).
- The exact server-side DINOv3 checkpoint and its metrics are not in this subsystem (the page comments mention val_boundary_iou=0.34 on 'v4 ep5' and a 13 MB ONNX + 2.3 GB external data, but the authoritative model files live in modal_inference/ or cloudrun_inference/).
- best_unetpp_v2.pth is named 'unetpp' but is actually SegFormer-B3 per convert_to_onnx.py; I trusted the conversion script over the filename but did not load the .pth to confirm its arch tag directly.
- README states the original UNet++ used EfficientNet-B7 encoder and 800 images (680/120 split); I could not cross-check those numbers against any training script in this subsystem.