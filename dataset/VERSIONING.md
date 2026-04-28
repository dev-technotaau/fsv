# Dataset Versioning & Reproducibility

**Why this matters for a client project**: when you ship model v1.2 and the client asks "what data was this trained on?", you must be able to answer precisely — filesystem state at time T, down to the byte. If training data drifts silently, every metric becomes non-reproducible and you can't audit regressions.

This document defines the versioning strategy for this repo's ML dataset artifacts.

---

## What gets versioned

There are **two kinds of files** in `dataset/` with very different versioning needs:

### Tier 1 — Small, git-committable (always tracked in git)

These are the **source-of-truth control files** — text-based, usually under 50 MB total. They define what the dataset *means* and how splits are partitioned. Commit to git directly.

```
dataset/manifest.jsonl                  (~20 MB)    which images exist + metadata
dataset/manifest_hq.jsonl               (~13 MB)    HQ subset manifest
dataset/integrity.json                  (~5 KB)     prepare_dataset.py audit
dataset/removed.jsonl                   (~50 KB)    what we deleted + why
dataset/resolution_report.json          (~10 KB)    tier distribution
dataset/licenses_per_source.json        (~5 KB)     source license breakdown
dataset/splits/train.jsonl              (~15 MB)
dataset/splits/val.jsonl                (~3 MB)
dataset/splits/test.jsonl               (~3 MB)     [read-only]
dataset/splits/train_hq.jsonl           (~9 MB)
dataset/splits/val_hq.jsonl             (~2 MB)
dataset/splits/test_hq.jsonl            (~2 MB)     [read-only]
dataset/splits/split_info.json          (~15 KB)    seed, hash, audit metadata
dataset/splits/split_dataset.log        (~5 KB)     run log
dataset/golden_set/manifest.jsonl       (~80 KB)
dataset/golden_set/selection_info.json  (~5 KB)
dataset/*.md                                        all documentation
```

### Tier 2 — Large binaries (DVC-tracked, NOT in git)

Image files themselves and their mask outputs. Git-LFS works but DVC is preferred for ML artifacts because it integrates with pipelines and has better remote-storage integration.

```
data_scraped/images/           ~21,665 JPGs        ~15 GB
data_scraped_neg/images/       ~12,009 JPGs        ~8 GB
data_scraped/rejected/         ~336 JPGs           ~200 MB
data_scraped_neg/rejected/     ~251 JPGs           ~150 MB
dataset/golden_set/images/     (optional copies)
dataset/golden_set/masks/      (once annotated)    ~5 MB PNGs
```

---

## Setup

### 1. Initialize DVC (once per repo clone)

```bash
pip install dvc dvc-s3   # or dvc-gdrive, dvc-azure, dvc-gcs — pick your remote
dvc init
git add .dvc/ .dvcignore
git commit -m "Initialize DVC"
```

### 2. Configure remote storage (one-time, pick ONE)

Client project → usually S3 / GCS / Azure Blob:
```bash
dvc remote add -d origin s3://client-ninja-fence-data/dvc-store
# then set AWS credentials via `aws configure` or env vars
```

Or a shared drive / NAS for simpler setups:
```bash
dvc remote add -d origin /mnt/nas/ninja-fence-dvc
```

### 3. Track the large image directories

```bash
dvc add data_scraped/images
dvc add data_scraped_neg/images
# This creates .dvc files (small — commit to git) and adds the real
# directories to .gitignore (so git won't try to version them).

git add data_scraped/images.dvc data_scraped_neg/images.dvc .gitignore
git commit -m "Track images via DVC"
```

### 4. Push to remote

```bash
dvc push
# Uploads the binary files to your configured remote.
```

---

## Day-to-day workflow

### When the scrape adds new images

```bash
# Re-scrape / re-run prepare_dataset.py + split_dataset.py
# ...

dvc add data_scraped/images    # DVC detects changes, updates checksums
git add data_scraped/images.dvc
git add dataset/manifest.jsonl dataset/splits/*.jsonl dataset/*.json
git commit -m "Dataset v1.1: added 2k hard-negatives, re-split seed=42"

dvc push
git push
```

### When cloning onto a new machine

```bash
git clone <repo>
cd <repo>
dvc pull              # downloads all referenced image files from remote
```

### When reproducing a historical experiment

```bash
git checkout <commit-sha>
dvc checkout         # syncs files to match this historical commit
# Now all JPGs on disk match the manifest.jsonl at that commit
```

---

## Version tagging convention

Use SemVer-lite for dataset releases:

| Tag format | When to bump | Example |
|------------|-------------|---------|
| `dataset-vMAJOR.0` | Incompatible composition change (new/removed sources, major filter revision) | `dataset-v2.0` |
| `dataset-vX.MINOR` | Additive changes (more images, new subcategories) | `dataset-v1.1` |
| `dataset-vX.Y.PATCH` | Bug-fix, e.g. noticing a corrupt image, cleaning a mistake | `dataset-v1.0.1` |

Tag examples:
```bash
git tag -a dataset-v1.0 -m "Initial 33,423 image release (33k total, stratified splits)"
git tag -a dataset-v1.1 -m "Added 2k hard negatives; re-split; golden-set added"
git push origin --tags
```

Each trained model checkpoint should record which dataset version it was trained on:
```yaml
# models/segformer-b5-v2.yaml (example)
dataset_version: dataset-v1.0
dataset_manifest_sha256: f2b0131993d9ea110e4e84999ebe8eabb800e2f2b9c4c3389436b8eee50a247b
split_info_sha256: <hash of split_info.json at train time>
```

The `manifest_sha256` recorded in `split_info.json` is your tamper-detection anchor. If someone modifies `manifest.jsonl` without re-running `split_dataset.py`, the next run will record a different hash — that's your audit signal.

---

## .dvcignore

Already configured in `.dvcignore` to skip temp / dev files.

---

## Disaster recovery

**Scenario: `data_scraped/images/` is accidentally wiped.**

1. `dvc pull data_scraped/images.dvc` — restore from DVC remote
2. Verify `dataset/manifest.jsonl` matches by running `prepare_dataset.py --dry-run` (it reports orphan files / metadata)

**Scenario: DVC remote is unreachable.**

Re-scrape is always possible — every row in `manifest.jsonl` has `origin_url` field. Rebuilding is slow (hours) but not lost.

**Scenario: Client needs training artifacts for compliance audit.**

1. `git checkout dataset-vX.Y` (the tag the model was trained against)
2. `dvc pull`
3. You now have the exact dataset state used for training.
4. Hand over the tagged commit + `dataset/` folder + `LICENSE_AUDIT.md` + `DATASHEET.md`.

---

## What NOT to commit

- **API keys / credentials** — already in `.gitignore`. Keep it that way.
- **`data_scraped/images/` directly** — it's 15+ GB of binaries. Use DVC.
- **Intermediate scrape state** — `data_scraped/scraper.log.jsonl`, `dedup.sqlite` — these are generated. Can be regenerated; don't bloat git history.
- **`.bak` backup files** — `prepare_dataset.py` and `standardize_images.py` create `.jsonl.bak` safety copies; don't commit them (`.gitignore` handles this).
