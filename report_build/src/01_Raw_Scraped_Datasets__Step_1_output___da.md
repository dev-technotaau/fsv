# AREA: Raw Scraped Datasets (Step 1 output): data_scraped/ and data_scraped_neg/

## SUMMARY
The repo ships two on-disk scrape corpora produced by Step 1 of the pipeline: a positive (fence) set in data_scraped/ and a negative (non-fence) hard-negative set in data_scraped_neg/. The positive set contains 21,414 images on disk drawn from 9 sources (Pexels, Playwright-scraped Bing/Google/Houzz/Pinterest, Wikimedia, Unsplash, Pixabay, and direct company sites), with a 21,674-line metadata.jsonl manifest; the negative set contains 12,009 images from 7 sources (nature/landscape/wood-texture distractors) with a matching 12,009-line manifest. Each manifest record is a JSON object with a fixed 12-field schema (path, source, query, origin_url, origin_page, title, sha256, dhash, width, height, bytes, extra). License info is captured only for the 3,519 Wikimedia images (inside extra). data_scraped/rejected/ holds 336 quarantined images, vision_qa_processed.txt is a 3,200-entry checkpoint of files already run through the Gemini-Vision QA filter, dedup.sqlite is the scraper's dedup/resume database, and metadata.jsonl.bak is a stale pre-rejection backup.

## KEY_FACTS
- data_scraped/images/ contains exactly 21,414 image files on disk (find ... | wc -l).
- data_scraped_neg/images/ contains exactly 12,009 image files on disk.
- Combined raw scrape total = 33,423 images (21,414 positive + 12,009 negative).
- data_scraped/metadata.jsonl has 21,674 lines (260 more than images on disk).
- data_scraped_neg/metadata.jsonl has 12,009 lines, exactly matching its 12,009 images on disk.
- Metadata schema is a 12-key JSON object per line: path, source, query, origin_url, origin_page, title, sha256, dhash, width, height, bytes, extra (verified via json.loads on first line).
- 'extra' is source-specific: Pexels carries photographer + native width/height; Unsplash carries native width/height + likes; Wikimedia carries mime, license, vision_label, vision_conf, vision_checked.
- Positive source distribution: pexels 5,166; pw_bing 4,166; wikimedia 3,519; pw_google 3,038; pw_houzz 1,859; unsplash 1,160; pixabay 1,014; company_sites 935; pw_pinterest 817.
- Negative source distribution: pw_google 2,740; pw_houzz 2,545; pexels 1,759; unsplash 1,654; pixabay 1,629; pw_pinterest 1,150; pw_bing 532.
- Positive queries are fence-centric: top queries 'cedar fence' (854), 'cedar fence residential backyard' (503), 'cedar privacy fence' (439), 'cedar horizontal slat fence' (423), 'bamboo fence' (418); query 'www.illinoisfencing.com' (381) shows company-site domains used as queries.
- Negative queries are non-fence distractors: 'forest hiking trail pine' (1,420), 'mountain landscape sunrise fog' (1,365), 'beach sunset ocean waves' (1,181), plus deliberate hard negatives like 'wooden lattice panel decorative wall' (142), 'wooden louvered door closet' (123), 'wooden horizontal blinds window' (93), 'log cabin wall exterior rustic' (80).
- License field present on exactly 3,519 metadata records, and ALL 3,519 are source=wikimedia (no license metadata captured for pexels/unsplash/pixabay/company/playwright sources).
- Wikimedia license values include CC BY 2.0 (182), CC BY 4.0 (117), CC BY 3.0 (64), Public domain, Attribution, plus localized variants (CC BY 2.5 dk/au, 3.0 de/us).
- data_scraped/rejected/ contains 336 images: 335 with pexels__ prefix and 1 company_sites__ prefix (quarantined, not in the main corpus).
- metadata.jsonl.bak has 22,010 lines vs current 21,674 -> diff is exactly 336, equal to the rejected count; the .bak is the pre-removal snapshot taken before 336 rejects were pruned.
- vision_qa_processed.txt has 3,200 lines, all unique (3,200 unique), a resume/checkpoint ledger of filenames already evaluated by the Gemini-Vision QA stage; 927 of the 3,200 are company_sites__ entries.
- The 336 rejected files do NOT appear in vision_qa_processed.txt (comm -12 overlap = 0), so rejection came from a different filter (e.g. dedup/format) than the vision QA pass.
- data_scraped/dedup.sqlite is 45,838,336 bytes (~45.8 MB) with tables: images (21,518 rows), urls_seen (92,091), failures (43,711), query_progress (3,117), sqlite_sequence.
- data_scraped_neg/dedup.sqlite is 14,630,912 bytes (~14.6 MB) with tables: images (12,009), urls_seen (25,828), failures (9,679), query_progress (841).
- scraper.log.jsonl (positive) has 23,777 lines; structured JSON with ts/level/logger/msg fields. Top msg types: slow_pipeline (18,300), progress (3,588), rate_limited (883), task_state (769); 16 run_start / 4 run_done events show ~16 scraper runs.
- scraper.log shows Gemini integration: msg 'gemini_expansion_ok' model 'gemini-2.5-flash' n=150 (query expansion), plus gemini_expansion_ok_vertex (5) and a vision_init_failed warning (missing GCP service-account JSON at D:\Web Devlopment\gen-lang-client-...json).
- scraper.log.jsonl (negative) has 4,493 lines.
- Filenames encode provenance: <source>__<query_slug>__<8hex>.jpg (e.g. pexels__cedar_fence__f807f793.jpg), where the 8 hex chars are the leading bytes of the sha256.
- Count reconciliation (positive): metadata 21,674 lines, sqlite images table 21,518 rows, images on disk 21,414 -> these three differ (manifest > db > disk), indicating drift between the ledgers and the actual files.

## FILE_ROLES
- [current] data_scraped/images/ — 21,414 raw scraped POSITIVE (fence) JPGs, Step 1 output, input to labeling/QA pipeline
- [current] data_scraped/metadata.jsonl — 21,674-line per-image provenance manifest (12-field schema) for the positive corpus
- [backup] data_scraped/metadata.jsonl.bak — Pre-pruning backup of the manifest, 22,010 lines = current + 336 rejected; superseded snapshot
- [artifact] data_scraped/rejected/ — 336 quarantined images (335 pexels + 1 company_sites) removed from the corpus
- [current] data_scraped/vision_qa_processed.txt — 3,200-entry resume checkpoint listing filenames already run through Gemini-Vision QA
- [current] data_scraped/dedup.sqlite — ~45.8MB scraper dedup/resume DB (images, urls_seen, failures, query_progress)
- [artifact] data_scraped/scraper.log.jsonl — 23,777-line structured run log of the positive scrape (~16 runs)
- [current] data_scraped_neg/images/ — 12,009 raw scraped NEGATIVE (non-fence) JPGs, hard-negative set for Step 1
- [current] data_scraped_neg/metadata.jsonl — 12,009-line provenance manifest for the negative corpus (matches image count exactly)
- [current] data_scraped_neg/dedup.sqlite — ~14.6MB scraper dedup/resume DB for the negative scrape
- [artifact] data_scraped_neg/scraper.log.jsonl — 4,493-line structured run log for the negative scrape

## NARRATIVE
## Step 1 on disk: the raw scrape corpora

Step 1 of the pipeline is a web scraper that builds two image corpora and lands them under `data_scraped/` (the positive, fence-bearing set) and `data_scraped_neg/` (the negative, deliberately fence-free set). Both directories follow the same internal layout: an `images/` folder of JPGs, a line-delimited JSON manifest (`metadata.jsonl`), a SQLite dedup/resume database (`dedup.sqlite`), and a structured run log (`scraper.log.jsonl`). The positive side additionally carries a `rejected/` quarantine folder, a `vision_qa_processed.txt` checkpoint, and a `.bak` of the manifest.

### How big is it, really

On disk the positive set holds **21,414** image files and the negative set holds **12,009**, for a combined raw harvest of **33,423** images. Those are hard `find ... -type f | wc -l` counts, not manifest estimates. The negative side is internally consistent — 12,009 images on disk, 12,009 manifest lines, and 12,009 rows in the sqlite `images` table all agree. The positive side does not line up so cleanly: the manifest has **21,674** lines, the sqlite `images` table reports **21,518** rows, and only **21,414** files actually sit on disk. So the manifest over-reports the disk by 260 and the database sits in between. This is the normal kind of drift you get when files are pruned or moved after the manifest/db were last written; it is worth noting in the report because any downstream consumer that trusts `metadata.jsonl` blindly will reference ~260 paths that no longer exist.

### What each manifest record looks like

Every line in `metadata.jsonl` is a self-contained JSON object with a stable 12-key schema: `path`, `source`, `query`, `origin_url`, `origin_page`, `title`, `sha256`, `dhash`, `width`, `height`, `bytes`, and `extra`. The first nine of those are flat and consistent across all sources; `extra` is a free-form nested object whose contents vary by provider. Pexels records tuck the photographer name and the original full-resolution dimensions into `extra`; Unsplash records carry the native dimensions plus a `likes` count; and Wikimedia records are the richest — they include `mime`, a `license` string, and three vision-QA fields (`vision_label`, `vision_conf`, `vision_checked`), e.g. `vision_label: "Wood"`, `vision_conf: 0.97`, `vision_checked: true`. The `path` field uses Windows-style backslashes (`data_scraped\\images\\...`), and `width`/`height` at the top level are the stored (downscaled) dimensions while `extra.width`/`extra.height` are the original source dimensions.

Filenames themselves encode provenance: the pattern is `<source>__<query_slug>__<8hexchars>.jpg`, for example `pexels__cedar_fence__f807f793.jpg`, where the eight hex characters are the leading bytes of the file's sha256. That makes every file traceable back to its source and search query without even opening the manifest.

### Where the images came from

The positive corpus pulls from **nine** sources. Pexels dominates at 5,166 images, followed by Playwright-driven Bing scraping (`pw_bing`, 4,166), Wikimedia (3,519), Playwright Google (`pw_google`, 3,038), Playwright Houzz (`pw_houzz`, 1,859), Unsplash (1,160), Pixabay (1,014), direct company websites (`company_sites`, 935), and Playwright Pinterest (`pw_pinterest`, 817). The `pw_` prefix denotes images harvested by browser automation rather than an official API, which matters for licensing diligence. The query distribution is exactly what you would want for a fence-staining model: the top queries are "cedar fence" (854), "cedar fence residential backyard" (503), "cedar privacy fence" (439), "cedar horizontal slat fence" (423), and "bamboo fence" (418). A long tail covers nearly every fence style (split rail, picket, shadowbox, board-on-board, stockade, dog-ear, vinyl, wrought iron, chain link) and contextual scenes (fence with pergola, with deck, around garden beds). Interestingly, some "queries" are actually company domains like `www.illinoisfencing.com` (381) — those correspond to the `company_sites` source, where the domain itself was the crawl seed.

The negative corpus is the mirror image: **seven** sources, led by `pw_google` (2,740) and `pw_houzz` (2,545), then Pexels (1,759), Unsplash (1,654), Pixabay (1,629), Pinterest (1,150), and Bing (532). Its queries are clearly engineered as distractors and hard negatives. The bulk are nature scenes that should never trip a fence detector — "forest hiking trail pine" (1,420), "mountain landscape sunrise fog" (1,365), "beach sunset ocean waves" (1,181), "desert sand dunes sky" (860). But the more telling entries are the deliberate hard negatives: "wooden lattice panel decorative wall" (142), "wooden louvered door closet" (123), "wooden horizontal blinds window" (93), "wooden plank flooring herringbone" (85), "log cabin wall exterior rustic" (80), "wooden deck boards close up floor" (63). Whoever built this set understood that the model's real failure mode is confusing fences with other repetitive wooden-slat structures, and seeded the negatives accordingly.

### Licensing

This is a soft spot worth flagging to the client. The `license` field appears on exactly **3,519** records, and every single one of them is a Wikimedia image. The license values are the expected Creative Commons spread — CC BY 2.0 (182), CC BY 4.0 (117), CC BY 3.0 (64), several localized CC BY variants, plus "Public domain" and bare "Attribution". For the other ~18,000 positive images (Pexels, Unsplash, Pixabay, company sites, and the Playwright-scraped Bing/Google/Houzz/Pinterest content) the manifest captures `origin_url` and `origin_page` but no explicit license string. The Pexels/Unsplash/Pixabay images fall under those platforms' permissive licenses, but the Playwright-scraped material (`pw_*` sources, ~9,880 positive images) has no recorded license at all, which is the usual gray area for browser-scraped image data.

### Rejected, QA, and the backup

`data_scraped/rejected/` holds **336** images — 335 with the `pexels__` prefix and a single `company_sites__` file — that were pulled out of the working corpus. The smoking gun for the `.bak` file is here: `metadata.jsonl.bak` has **22,010** lines versus the current **21,674**, a difference of exactly **336**. So the `.bak` is the manifest snapshot taken before those 336 rejects were pruned; it is a backup/superseded artifact, not a live file. Notably, none of the 336 rejected files appear in `vision_qa_processed.txt` (set intersection is zero), which tells us they were rejected by a different filter — likely dedup or format/decode validation — rather than by the vision-QA content check.

`vision_qa_processed.txt` is a **3,200**-line (all unique) resume checkpoint. It is simply a flat list of filenames the scraper has already pushed through its Gemini-Vision QA stage, so a re-run can skip them; 927 of the 3,200 are `company_sites__` images, consistent with company-site scrapes being the noisiest source that most needed content verification. The vision QA itself is wired to Google Gemini — the run log shows `gemini_expansion_ok` with model `gemini-2.5-flash` (query expansion, n=150) and a `vision_init_failed` warning about a missing GCP service-account JSON, confirming the QA/vision path depends on a Gemini/Vertex credential that wasn't always present.

### The bookkeeping databases and logs

`dedup.sqlite` is the scraper's brain for dedup and resumability. The positive DB is ~45.8 MB with five tables: `images` (21,518 rows), `urls_seen` (92,091 — every URL the scraper has touched), `failures` (43,711 — failed/skipped downloads), `query_progress` (3,117), and the sqlite housekeeping table. The negative DB is ~14.6 MB: `images` (12,009), `urls_seen` (25,828), `failures` (9,679), `query_progress` (841). The very high `failures` counts (43,711 positive, 9,679 negative) relative to kept images show how aggressively the scraper filtered — most URLs it encountered were rejected, deduped, or failed to download. The `scraper.log.jsonl` files (23,777 lines positive, 4,493 negative) are structured JSONL with `ts`/`level`/`logger`/`msg`. The dominant message is `slow_pipeline` (18,300 occurrences), a throughput warning, followed by `progress` (3,588) and `rate_limited` (883). There are 16 `run_start` events, confirming the positive corpus was assembled over roughly sixteen separate scraper sessions.

### Classification

`data_scraped/images/`, `data_scraped_neg/images/`, both `metadata.jsonl` files, `vision_qa_processed.txt`, and both `dedup.sqlite` files are **CURRENT** — they are the live Step 1 output and resume state. `metadata.jsonl.bak` is a **BACKUP** (pre-rejection snapshot, +336 lines). `data_scraped/rejected/` and both `scraper.log.jsonl` files are **ARTIFACTS** — quarantine and run logs, not consumed downstream.

## UNCERTAINTIES
- The positive-side counts do not reconcile: metadata.jsonl=21,674 lines, sqlite images=21,518 rows, images on disk=21,414. I confirmed the numbers but cannot determine from the data alone which is authoritative or what process caused the 260-line manifest-vs-disk gap.
- License coverage: only Wikimedia images carry an explicit license field. I cannot verify the actual licensing/usage rights of the ~9,880 Playwright-scraped (pw_*) positive images or the company_sites images from the manifest alone; this is a legal/compliance question for the client.
- vision_qa_processed.txt holds 3,200 entries while the positive corpus is ~21,400 images, so the vision-QA pass appears to have covered only a subset (heavily company_sites). I cannot confirm whether the remaining images were QA'd by another mechanism or skipped.
- I did not locate or read the scraper source code itself (Step 1 script) in this audit — these conclusions are inferred purely from the on-disk outputs (manifests, logs, sqlite). The exact rejection criteria and what 'slow_pipeline' precisely measures are inferred, not code-verified.
- scraper.log references a Gemini service-account JSON at 'D:\Web Devlopment\gen-lang-client-...json' that 'was not found'; whether vision QA ran successfully in later runs (when credentials were present) vs. was silently skipped could not be fully determined from the log sample.
- The 'extra' object schema is verified for Pexels, Unsplash, and Wikimedia only; I did not exhaustively verify the extra-field contents for pixabay, company_sites, and the pw_* sources.