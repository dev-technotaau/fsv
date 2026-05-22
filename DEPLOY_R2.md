# Deploy the F-Stain model to Cloudflare R2

A ~20-minute setup that puts `fence_model_dinov2.onnx` behind Cloudflare's
global edge — free, no egress fees, and ~2-3× faster first-load than the
current Hugging Face Space hosting.

---

## What you're doing

| Today | After |
|---|---|
| Model served from Hugging Face Space static CDN (a few regions) | Model served from R2 via Cloudflare's global edge (~100 PoPs) |
| Each cold visit fully re-downloads 135 MB | First visit downloads once, cached forever (immutable header) |
| One URL to maintain | Same — just point `MODEL_PATH` at the new URL |

You keep the Hugging Face Space as the **app** (the HTML page itself).
Only the heavy ONNX file moves to R2.

---

## Prerequisites

1. A Cloudflare account (free tier — no credit card needed for R2 within free limits)
2. The model file: `fence-staining-visualizer/fence_model_dinov2.onnx` (~135 MB)
3. Optional but recommended: a domain you own (for a clean URL like
   `models.your-domain.com`). If you don't have one, R2's `r2.dev` URL works
   for testing — see Step 5 for both paths.

---

## Step 1 — Create the R2 bucket

1. Log in to **dash.cloudflare.com**
2. Sidebar → **R2 Object Storage** → click **Get Started** if it's your first time
3. Click **Create bucket**
4. Bucket name: `f-stain-models` (or any lowercase-with-hyphens name)
5. Location: **Automatic** (Cloudflare picks the optimal region)
6. Default Storage Class: **Standard**
7. Click **Create bucket**

You now have an empty bucket. No charges until you store/serve from it,
and the free tier covers everything you'll do.

---

## Step 2 — Upload the model

Two options. **Option B (wrangler) is recommended** because it sets
the `Cache-Control` header atomically with the upload.

### Option A — Dashboard upload (simplest)

1. Open the bucket → **Objects** tab → **Upload**
2. Drag-and-drop `fence_model_dinov2.onnx`
3. Wait for the 135 MB upload to finish (~1-3 min depending on your line)
4. After upload, click the file → **Edit metadata** → add:
   - `Cache-Control` = `public, max-age=31536000, immutable`
   - `Content-Type` = `application/octet-stream`

### Option B — wrangler CLI (recommended)

One-time setup:

```bash
npm install -g wrangler
wrangler login           # opens browser → authorize
```

Then upload with the correct headers in one shot:

```bash
cd fence-staining-visualizer
wrangler r2 object put f-stain-models/fence_model_dinov2.onnx \
  --file=fence_model_dinov2.onnx \
  --content-type=application/octet-stream \
  --cache-control="public, max-age=31536000, immutable"
```

The `Cache-Control` line is the most important part — it tells the browser
"this file never changes, cache it forever". Returning visitors will load
the model from disk cache (0 ms) instead of redownloading.

---

## Step 3 — Configure CORS

Without this, the browser will block fetching the model because the page
(HF Space) and the model (R2) are on different domains.

1. R2 dashboard → your bucket → **Settings** → **CORS Policy** → **Add**
2. Paste this JSON:

```json
[
  {
    "AllowedOrigins": [
      "https://dev-technotaau-f-stain.static.hf.space",
      "https://huggingface.co",
      "http://localhost:8000",
      "http://127.0.0.1:8000"
    ],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag", "Content-Length", "Content-Type"],
    "MaxAgeSeconds": 86400
  }
]
```

Replace / extend `AllowedOrigins` with any other domain you'll serve the
HTML from. The `localhost` entries let you test the deployed model from a
local dev server.

> **Don't use `"*"` for AllowedOrigins** — it works, but it's a public
> bucket fingerprint your URL ends up shared. Listing only your real
> origins keeps casual hotlinking down.

---

## Step 4 — Enable public access

R2 buckets are private by default. You have two ways to expose the file:

### Option A — Connect a custom domain (production-grade, recommended)

1. R2 dashboard → bucket → **Settings** → **Public access** → **Connect Domain**
2. Enter a subdomain you control (e.g. `models.your-domain.com`)
3. Cloudflare auto-creates the DNS record (your domain must already be on
   Cloudflare DNS — if not, transfer DNS first, takes ~5 min)
4. SSL provisions automatically (~1-2 min)

Your model URL becomes: `https://models.your-domain.com/fence_model_dinov2.onnx`

This URL is permanent, brand-aligned, and has no rate limit beyond R2's
free-tier quotas.

### Option B — Enable `r2.dev` subdomain (quick start, dev only)

1. R2 dashboard → bucket → **Settings** → **Public access** → **Allow Access**
   on the **R2.dev Subdomain** row
2. Note the URL Cloudflare gives you, e.g.
   `https://pub-abc123def456.r2.dev`

Your model URL becomes:
`https://pub-abc123def456.r2.dev/fence_model_dinov2.onnx`

> **Cloudflare rate-limits `r2.dev`** for development use — it can throttle
> or 429 production traffic. Use it for testing only, then move to a custom
> domain before showing the client.

---

## Step 5 — Update the web app

Open [fence-staining-visualizer/index2.html](fence-staining-visualizer/index2.html)
and find this line in the `CONFIG` block (around line 1035):

```javascript
MODEL_PATH: './fence_model_dinov2.onnx',
```

Change it to your R2 URL:

```javascript
MODEL_PATH: 'https://models.your-domain.com/fence_model_dinov2.onnx',
```

That's the only code change.

Re-deploy [index2.html](fence-staining-visualizer/index2.html) to the
Hugging Face Space. The page now loads the model from R2 instead of HF.

---

## Step 6 — Verify

Open the deployed page in **Chrome DevTools → Network tab**, hard-refresh
(Ctrl+Shift+R), and watch for `fence_model_dinov2.onnx`. You should see:

| Check | Expected |
|---|---|
| **Status** | 200 (first load) or 200 (from disk cache) on reload |
| **Time** | Much faster than before (typically 3-10s for first load) |
| **Response Headers → cache-control** | `public, max-age=31536000, immutable` |
| **Response Headers → cf-cache-status** | `HIT` after the second visitor anywhere |
| **Response Headers → server** | `cloudflare` |
| **Console** | No CORS errors |

If you see a CORS error, double-check Step 3 — the page's origin must be
in `AllowedOrigins`.

If `cf-cache-status` stays `MISS` after multiple visits from the same
region, the cache headers didn't apply — re-check the upload in Step 2.

---

## Free tier limits (you won't hit them)

| Limit | Free tier | Your usage at 10,000 first-time visitors/month |
|---|---|---|
| Storage | 10 GB | 0.135 GB |
| Class A operations (PUT/POST) | 1M/mo | < 100 (only when re-uploading) |
| Class B operations (GET) | 10M/mo | ~10K (cache absorbs the rest) |
| **Egress bandwidth** | **Unlimited & free** | n/a — this is R2's killer feature |

Even if 1 million unique users load the page in a month, you stay free
because Cloudflare's edge cache means R2 itself only sees a few thousand
origin requests.

---

## Rollback

If anything goes wrong, revert is one line — change `MODEL_PATH` in
[index2.html](fence-staining-visualizer/index2.html) back to the relative
path `'./fence_model_dinov2.onnx'` and the HF Space serves it again.
Nothing in the R2 setup interferes with the existing HF deployment.

---

## What this does NOT change

- The model itself — same `fence_model_dinov2.onnx`, same predictions,
  same sidecar JSON
- The HTML page — still hosted on the HF Space
- The browser code — only `MODEL_PATH` changes
- User privacy — the model still runs locally in the browser; R2 only
  serves the static file, never sees user images

---

*Total time: ~20 min for custom-domain setup, ~5 min for `r2.dev` quick start.*
