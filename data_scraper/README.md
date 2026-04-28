# Fence image scraper

Async, multi-source image collector for training a multi-class fence segmentation model. Live SQLite-backed dedup, quality filtering, optional Google Vision verification.

## Sources (10 total, running concurrently)

**Paid (best quality, lowest rate, require API keys):**
- `google_cse` — Google Custom Search Image API (100 free queries/day, then $5/1000)
- `google_vision` — **verifier only**, not a scraping source. Confirms fence label after download.

**Free with API key (high quality):**
- `pexels` — Pexels API (200 req/hr)
- `unsplash` — Unsplash API (50 req/hr demo tier)
- `pixabay` — Pixabay API (100 req/60s)
- `flickr` — Flickr API (CC-licensed only)

**Free without key:**
- `wikimedia` — Wikimedia Commons
- `reddit` — public JSON endpoints, configurable subreddits

**Free via Playwright (ToS-risky):**
- `pw_google` — Google Images (**disabled by default** — violates ToS; use CSE instead)
- `pw_bing` — Bing Images (disabled by default)
- `pw_ddg` — DuckDuckGo Images (enabled — DDG is more permissive)

## Architecture

```
queries.py ─┐
            ├─► sources/ (async generators, run concurrently)
            ▼
  [asyncio.Queue (bounded)]
            ▼
  N download workers
     │
     ├─► fetch bytes (async httpx, retry + backoff)
     ├─► SHA256 dedup  ─┐
     ├─► quality check  │   SQLite (urls_seen, images, failures)
     ├─► dHash near-dup ┤
     ├─► (optional) Google Vision verify
     └─► save image + metadata JSONL
```

Live dedup means the SAME image scraped by different sources is saved once. SQLite WAL mode keeps concurrent reads/writes safe.

## Install

```bash
pip install -r requirements/scraper.txt
playwright install chromium      # only if you want Playwright sources
```

## API keys — set as env vars

```bash
# Paid
export GOOGLE_CSE_API_KEY=...
export GOOGLE_CSE_CX=...                          # Custom Search Engine ID
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json   # for Vision

# Free tiers
export PEXELS_API_KEY=...
export UNSPLASH_ACCESS_KEY=...
export PIXABAY_API_KEY=...
export FLICKR_API_KEY=...

# Optional query expansion
export GEMINI_API_KEY=...
```

Obtaining keys:
- **Google CSE**: https://programmablesearchengine.google.com/ (create engine, enable image search) + https://developers.google.com/custom-search/v1/introduction
- **Google Vision**: https://console.cloud.google.com/apis/library/vision.googleapis.com (service account JSON)
- **Pexels**: https://www.pexels.com/api/
- **Unsplash**: https://unsplash.com/developers
- **Pixabay**: https://pixabay.com/api/docs/
- **Flickr**: https://www.flickr.com/services/api/misc.api_keys.html

## Quick start

```bash
# 1. Check config + credentials
python -m data_scraper.cli preflight

# 2. Preview queries (200+ static + optional Gemini expansion)
python -m data_scraper.cli queries

# 3. Run the scrape
python -m data_scraper.cli run

# Override any config scalar via --set
python -m data_scraper.cli run \
    --set runtime.target_total_images=15000 \
    --set runtime.download_workers=24 \
    --set google_vision.enabled=true

# 4. Check progress / stats anytime
python -m data_scraper.cli stats

# 5. Export metadata as CSV
python -m data_scraper.cli export -o data_scraped/metadata.csv
```

The scraper is **resumable**. If killed with Ctrl+C (or crashed), re-running uses the SQLite dedup to skip already-scraped URLs and images.

## Output layout

```
data_scraped/
├── dedup.sqlite               # state: url history, image hashes, failures
├── metadata.jsonl             # one line per saved image with all metadata
├── failures.jsonl             # one line per rejected candidate (reason included)
└── images/
    ├── pexels__cedar_fence__a1b2c3d4.jpg
    ├── google_cse__wooden_privacy_fence_with_tree_branches__e5f6g7h8.jpg
    └── ...
```

Filenames encode `<source>__<slugified_query>__<sha256[:8]>.jpg` for traceability.

## Query corpus

`queries.py` ships **~200 static queries** organized into categories designed for a multi-class occlusion-aware fence model:

- **A. Wood types & styles** — cedar, redwood, pine, pressure-treated, shadowbox, picket, horizontal slat, split rail, ...
- **B. Scenes/environments** — backyard, front yard, garden, patio, pool area, property line, ...
- **C. Occlusions** (critical for your training goals) — fence with tree branches in front, behind bushes, with climbing vines, partially hidden by plants, covered in ivy, ...
- **D. Animals/humans near fence** — dog behind/in front of fence, cat on fence, person painting, installing, gardening, ...
- **E. Similar-material distractors** — wooden shed, house siding, pergola, telephone pole, deck, trellis, gazebo, log cabin, ...
- **F. Scale variety** — close-up texture, macro, aerial view, distant fence, long perspective, ...
- **G. Weather/time-of-day** — golden hour, sunset, rain, frost, snow, fog, dramatic shadows, ...
- **H. Structural variants** — broken/rotting fence, gate, missing board, partial fence, corner, new construction, ...

Enable `queries.use_gemini_expansion: true` + set `GEMINI_API_KEY` to add another 100-200 LLM-generated queries.

## Tuning for 10k–15k target

The YAML ships per-source `target_images` soft targets that sum to roughly 13,500. In practice you'll get 60-80% of that after quality filtering + dedup, so 8-12k clean images per run. If that's short:

- Raise `pexels` / `flickr` / `pixabay` targets — they have large catalogs and aren't rate-limited hard
- Enable `pw_bing` (ToS-grey — be careful)
- Add more queries via Gemini or the `queries.custom` list

## Google Vision verification

Flip `google_vision.enabled: true` in the config. Every downloaded image then gets Vision-API labeled; if no label in the accept list scores ≥ `min_fence_confidence`, the image is rejected.

Cost: ~$1.50 per 1000 images. Worth it for quality if you're paying for compute for training anyway.

## Troubleshooting

- **"playwright not installed"** — either `pip install playwright && playwright install chromium`, or set `pw_*.enabled: false`.
- **Rate limit 429s** — lower `rate_limit_per_minute` for the offending source. Adaptive rate limiter auto-respects `Retry-After` / `X-RateLimit-*` headers.
- **Disk fills up** — each image re-encoded to JPEG quality 92, ~200-500 KB each. 12k images ≈ 3-6 GB. Disk guard stops run below `runtime.min_free_disk_gb` (default 2.0).
- **Google CSE `userRateLimitExceeded`** — you hit the 100/day free tier. Wait 24h or upgrade.
- **Reddit 429s on large runs** — unauthenticated mode is ~10 req/min. Set `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` env vars for OAuth (60 req/min). See Reddit OAuth section below.

## Reddit OAuth (recommended)

Unauthenticated Reddit API: ~10 requests/minute per IP. OAuth raises this to 60/min per account.

**Setup (2 minutes):**

1. Visit https://www.reddit.com/prefs/apps and click **"create another app..."**
2. Choose type: **"script"**
3. Name: anything (e.g. `ninja-fence-scraper`)
4. Redirect URI: `http://localhost:8080` (required but unused)
5. After creation:
   - `client_id` = shown under the app name (14-char string)
   - `client_secret` = the "secret" field (27-char string)
6. Set env vars:
   ```bash
   export REDDIT_CLIENT_ID=abc123...
   export REDDIT_CLIENT_SECRET=xyz789...
   ```
7. Ensure `reddit.user_agent` in config identifies you (e.g. `"ninja-fence-scraper/0.1 by /u/yourusername"`) — Reddit bans generic agents.

The scraper uses the client-credentials flow (app-only auth, no Reddit account needed). Token auto-refreshes. If you want user-scope requests, also set `reddit.username` and `reddit.password` to use password flow.

## Google Vision cost control

Vision verification defaults to `sample_rate: 1.0` (checks every image, ~$1.50/1000). To cut cost while still catching off-topic drift, lower the sample rate:

| `sample_rate` | Coverage | Cost per 1000 scraped |
|---|---|---|
| 1.0  | 100% | $1.50 |
| 0.5  | 50%  | $0.75 |
| 0.2  | 20%  | $0.30 |
| 0.05 | 5%   | $0.075 |

Sampling is **deterministic per-image** (derived from SHA256), so restarts don't re-bill already-verified images. Non-sampled images are accepted without API call.

For a 12k-image run at `sample_rate=0.2`: ~2400 API calls ≈ $3.60 total. Reasonable insurance against the worst-case off-topic drift that can occur when pw_bing returns wallpaper-ish images.

## Google Images via Playwright — do NOT enable

`pw_google` scrapes google.com which **violates Google's Terms of Service**. It exists in the code only as emergency fallback. Use cases where it's wrong:

- Any production workflow — use `google_cse` with an API key
- Any case where you care about not getting IP-banned
- Any case where reproducibility matters (Google breaks the selectors regularly)

If you still want it: set `pw_google.enabled: true` in your config. The selectors in [playwright_google.py](sources/playwright_google.py) will probably need updating — Google A/B tests their image-results markup constantly. Your account/IP can get CAPTCHA-walled for days. You accept the risk.
