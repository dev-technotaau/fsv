# Deploy F-Stain inference to Modal

Server-side inference for the F-Stain web app. The model lives on Modal,
the browser only uploads photos and gets back a small PNG mask. First
result in ~2-3 seconds instead of waiting 22s for the model to download.

---

## Architecture

```
┌──────────────────────┐                  ┌─────────────────────────┐
│  Browser             │  multipart POST  │  Modal container        │
│  index2.html         │ ───────────────► │  ONNX inference         │
│  (HF Static Space)   │   ~200 KB image  │  (DINOv2-S, CPU)        │
│                      │ ◄─────────────── │                         │
│  Postprocess + UI    │  ~50 KB PNG mask │  Model loaded once,     │
│  (color, opacity)    │                  │  reused per request     │
└──────────────────────┘                  └─────────────────────────┘
```

- **Browser** stays static (HF Space). Only the *model loading* is replaced
  with an HTTP call to Modal.
- **Modal** runs the ONNX session on a 2-CPU container, scales to zero
  when idle (free), warms up in ~3-5s when called.
- **Mask** comes back as a 518×518 grayscale PNG (~30-100 KB). Browser
  decodes, converts to floats, runs the same soft-mask + CC cleanup +
  vegetation filter post-processing it always did.

---

## Prerequisites

1. A Modal account — sign up at **modal.com** (no credit card needed for
   the free $30/month credit).
2. Python 3.10+ on your machine.
3. The ONNX model already at:
   `fence-staining-visualizer/fence_model_dinov2.onnx`

---

## Step 1 — Install the Modal CLI

In your `ml` conda env (or any Python env):

```powershell
pip install modal
```

---

## Step 2 — Authenticate

One-time:

```powershell
modal token new
```

This opens a browser. Click **Authorize** to link the CLI to your Modal
workspace. The CLI stores a token in `~/.modal/config.toml` — you won't
need to run this again on the same machine.

---

## Step 3 — Deploy

From this folder:

```powershell
cd modal_inference
modal deploy app.py
```

The first deploy builds the container image (~2-3 minutes — installs ORT,
PIL, FastAPI, and uploads the 135 MB ONNX). Subsequent deploys reuse the
cached image layers and are seconds.

When it finishes you'll see something like:

```
✓ Created web => https://your-workspace--f-stain-inference-web.modal.run
```

**Copy that URL.** That's your inference endpoint.

---

## Step 4 — Test the endpoint

Quick health check (this also wakes a cold container):

```powershell
curl https://your-workspace--f-stain-inference-web.modal.run/
```

Expected response:

```json
{"status":"ok","service":"f-stain-inference","model_input_size":518,"channel_order":"RGB"}
```

Real inference test (replace path with any fence photo):

```powershell
curl -X POST `
  -F "image=@C:\path\to\fence.jpg" `
  https://your-workspace--f-stain-inference-web.modal.run/detect `
  --output mask.png
```

Open `mask.png` — should be a grayscale 518×518 fence mask. White = fence,
black = background, gray = uncertain edges.

---

## Step 5 — Point the frontend at Modal

Open [fence-staining-visualizer/index2.html](../fence-staining-visualizer/index2.html)
and find the `CONFIG` block (around line 1035). Change:

```javascript
MODAL_ENDPOINT: 'https://YOUR-WORKSPACE--f-stain-inference-web.modal.run/detect',
```

…to your real URL from Step 3 (with `/detect` at the end).

Re-deploy [index2.html](../fence-staining-visualizer/index2.html) to your
HF Space. Done.

---

## Verifying it works

Load the page in a browser, DevTools → Network tab:

| What you should see | Means |
|---|---|
| No `fence_model_dinov2.onnx` request at all | Browser no longer downloads the model — page loads instantly |
| First `/` GET (~100 ms) on page load | Container wake-up ping |
| `POST /detect` after clicking *Detect Fence*, ~1-5s | Inference (cold) or ~0.5-1.5s (warm) |
| `Content-Length` of the response: ~30-100 KB | Compact PNG mask |
| Response header `X-Inference-Ms: 800` | Server-side timing for debugging |

---

## Cost expectations

Modal CPU pricing is ~$0.000056/sec (~$0.20/CPU/hour). With the defaults
in `app.py`:

| Traffic | Containers needed | Est. monthly cost |
|---|---|---|
| 100 visitors / mo (~300 inferences) | 1, scales to zero | **< $0.01** |
| 5,000 visitors / mo (~15K inferences) | 1, scales to zero | **~$1-2** |
| 50,000 visitors / mo (~150K inferences) | 2-3, scales to zero | **~$8-15** |
| Always-warm (`min_containers=1`) | 1 alive 24/7 | **+~$10/mo** baseline |

Your $30 free credit covers tens of thousands of inferences per month
even *with* `min_containers=1` always-warm.

Check actual usage in the Modal dashboard at **modal.com → Usage**.

---

## Optional tuning

### Eliminate cold starts (small extra cost)

Default `min_containers=0` means the first visitor after 5 min idle waits
~3-5s for the container to start. To always keep one container warm,
edit `app.py`:

```python
min_containers=1,    # was 0
```

Adds ~$10/mo baseline but every user gets sub-second cold start.

### Use a GPU (overkill for DINOv2-S, but possible)

Add `gpu="T4"` to the `@app.function(...)` decorator. Drops inference
time from ~600ms to ~50ms but costs ~10× more per second. Not worth it
unless you serve > 100 inferences/minute.

### Restrict CORS to fewer origins

In `app.py`, edit `ALLOWED_ORIGINS` to just the host(s) your HTML lives
on. The default list includes localhost for dev.

---

## Updating the model

When you retrain and want to ship a new ONNX:

1. Replace `fence-staining-visualizer/fence_model_dinov2.onnx`
2. From `modal_inference/`, run: `modal deploy app.py`

The frontend URL stays the same. Modal redeploys in seconds (only the
model layer changes).

---

## Rolling back

If anything goes wrong on Modal, the rollback is one line in
`index2.html`:

```javascript
MODAL_ENDPOINT: '',   // empty disables Modal mode
```

…and uncomment the original `<script src="...ort.min.js"></script>` line
to fall back to client-side inference (you'll also need to revert the
`detectFence()` function — keep the previous version around for safety
during the transition).

For a quick "service down" message without code edits, you can also
**stop the Modal app** from the dashboard: Apps → f-stain-inference →
Stop. The browser will show a clean error message.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| CORS error in browser console | Your HTML's origin isn't in `ALLOWED_ORIGINS` | Edit `app.py`, redeploy |
| `404 Not Found` on POST | Wrong URL — missing `/detect` | Append `/detect` to MODAL_ENDPOINT |
| First request takes 30s+ | Modal image build still running | Wait, check `modal app logs f-stain-inference` |
| Mask comes back all black | Image was downscaled too aggressively client-side | Bump `imageToUploadBlob`'s maxDim from 1024 to 1600 in JS |
| "Empty upload" 400 | Browser didn't attach the file | Check Form name is `image` |
| Modal credits draining fast | `min_containers > 0` or runaway requests | Set `max_containers` lower, monitor dashboard |

To stream live container logs while testing:

```powershell
modal app logs f-stain-inference
```
