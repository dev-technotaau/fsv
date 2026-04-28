"""Pydantic config + YAML loader. Source API keys come from env or YAML."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ConfigDict


# ---------- per-source configs ----------

class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    target_images: int = 2000               # soft target per source (stops when reached)
    rate_limit_per_minute: int = 60
    max_concurrent: int = 4


class GoogleCSEConfig(SourceConfig):
    api_key: Optional[str] = None
    cx: Optional[str] = None                # Custom Search engine ID


class PexelsConfig(SourceConfig):
    api_key: Optional[str] = None


class UnsplashConfig(SourceConfig):
    access_key: Optional[str] = None


class PixabayConfig(SourceConfig):
    api_key: Optional[str] = None


class FlickrConfig(SourceConfig):
    api_key: Optional[str] = None


class RedditConfig(SourceConfig):
    subreddits: list[str] = Field(default_factory=lambda: [
        "Fencing", "Homestead", "Landscaping", "Gardening",
        "BackyardDesign", "LandscapingTips", "pics",
    ])
    user_agent: str = "ninja-fence-scraper/0.1 by /u/yourusername"
    # OAuth (optional but recommended — 60 req/min vs ~10 unauthenticated)
    # Create a "script" app at https://www.reddit.com/prefs/apps
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    # If username+password set, uses password flow (full user auth).
    # Otherwise uses client-credentials (app-only) flow.
    username: Optional[str] = None
    password: Optional[str] = None


class PlaywrightConfig(SourceConfig):
    headless: bool = True
    browser: Literal["chromium", "firefox"] = "chromium"
    max_scrolls: int = 40
    scroll_delay_s: float = 1.2
    request_timeout_s: int = 30


class CompanySitesConfig(SourceConfig):
    """List of direct company gallery URLs to scrape. Each URL = one fetch;
    no search or query expansion. Target smaller than search-based sources."""
    urls: list[str] = Field(default_factory=list)


class GoogleVisionConfig(BaseModel):
    """Google Cloud Vision is used as a VERIFIER after download,
    not as a scraping source itself.

    Cost: ~$1.50 per 1000 images at full sample rate. Set sample_rate < 1.0
    to check only a fraction (randomly sampled) and reduce cost linearly.
    """
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    credentials_json: Optional[str] = None  # path to service account json
    min_fence_confidence: float = 0.6       # reject image if no fence-related label ≥ this
    sample_rate: float = 1.0                # 1.0=verify all; 0.2=verify 20% (cost × 0.2)
    # Batch annotation — 10× fewer HTTP round-trips to Vision
    batch_size: int = 10                    # max 16 per Vision's batch_annotate_images
    batch_timeout_ms: int = 500             # flush partial batches after this many ms
    # When True, Vision fetches images server-side via URI (no upload bandwidth used).
    # ~10000× less local upload than sending bytes. Falls back to bytes if URI is unsupported.
    use_uri_mode: bool = True
    labels_to_accept: list[str] = Field(default_factory=lambda: [
        "Fence", "Picket fence", "Wood", "Home fencing", "Yard",
        "Wall", "Garden", "Landscape",
    ])


# ---------- quality / dedup / storage ----------

class QualityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_width: int = 800
    min_height: int = 600
    max_width: int = 8000
    max_height: int = 8000
    min_aspect: float = 0.4
    max_aspect: float = 2.5
    min_bytes: int = 40_000
    max_bytes: int = 25_000_000
    blocked_domains: list[str] = Field(default_factory=lambda: [
        "istockphoto.com", "shutterstock.com", "gettyimages.com",
        "alamy.com", "dreamstime.com", "123rf.com", "depositphotos.com",
        "stock.adobe.com", "adobestock.com",
    ])


class DedupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phash_hamming_threshold: int = 5        # ≤ this = duplicate
    db_path: Path = Path("data_scraped/dedup.sqlite")


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    images_dir: Path = Path("data_scraped/images")
    metadata_jsonl: Path = Path("data_scraped/metadata.jsonl")
    failed_log: Path = Path("data_scraped/failures.jsonl")
    file_extension: Literal["jpg", "png", "webp"] = "jpg"


class QueriesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    use_static: bool = True
    use_gemini_expansion: bool = False
    gemini_api_key: Optional[str] = None
    gemini_target_extra: int = 100          # how many extra queries to ask Gemini for
    custom: list[str] = Field(default_factory=list)  # user-added queries


# ---------- runtime ----------

class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_total_images: int = 12_000
    download_workers: int = 16
    download_timeout_s: int = 30
    max_retries: int = 3
    progress_interval_s: float = 2.0
    output_root: Path = Path("data_scraped")
    # new gap-fixes
    dry_run: bool = False
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
    log_file: Optional[Path] = Path("data_scraped/scraper.log.jsonl")
    log_format: Literal["plain", "json"] = "plain"
    min_free_disk_gb: float = 2.0
    disk_check_interval_s: float = 30.0
    max_image_megapixels: int = 100


class ProxyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    proxies: list[str] = Field(default_factory=list)
    failure_cool_down_s: float = 300.0


class CircuitBreakerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    failure_threshold: int = 5
    cool_down_s: float = 60.0


class ContentFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    extra_block_keywords: list[str] = Field(default_factory=list)
    extra_block_domains: list[str] = Field(default_factory=list)


class QueryPriorityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    target_per_query: int = 100


class DistributedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    redis_url: Optional[str] = None
    key_prefix: str = "fence_scraper:"


class BatchedWriterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    batch_size: int = 50
    flush_interval_s: float = 0.5


# ---------- top-level ----------

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    queries: QueriesConfig = Field(default_factory=QueriesConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    # new gap-fix sub-configs
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    content_filter: ContentFilterConfig = Field(default_factory=ContentFilterConfig)
    query_priority: QueryPriorityConfig = Field(default_factory=QueryPriorityConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    batched_writer: BatchedWriterConfig = Field(default_factory=BatchedWriterConfig)

    google_cse: GoogleCSEConfig = Field(default_factory=GoogleCSEConfig)
    google_vision: GoogleVisionConfig = Field(default_factory=GoogleVisionConfig)
    pexels: PexelsConfig = Field(default_factory=PexelsConfig)
    unsplash: UnsplashConfig = Field(default_factory=UnsplashConfig)
    pixabay: PixabayConfig = Field(default_factory=PixabayConfig)
    flickr: FlickrConfig = Field(default_factory=FlickrConfig)
    wikimedia: SourceConfig = Field(default_factory=SourceConfig)
    reddit: RedditConfig = Field(default_factory=RedditConfig)
    pw_google: PlaywrightConfig = Field(default_factory=PlaywrightConfig)
    pw_bing: PlaywrightConfig = Field(default_factory=PlaywrightConfig)
    pw_ddg: PlaywrightConfig = Field(default_factory=PlaywrightConfig)
    pw_pinterest: PlaywrightConfig = Field(default_factory=PlaywrightConfig)
    pw_houzz: PlaywrightConfig = Field(default_factory=PlaywrightConfig)
    company_sites: CompanySitesConfig = Field(default_factory=CompanySitesConfig)


# ---------- loader ----------

def _expand_env(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} in strings."""
    if isinstance(value, str) and "${" in value:
        import re
        def repl(m):
            return os.environ.get(m.group(1), m.group(0))
        return re.sub(r"\$\{([A-Z0-9_]+)\}", repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: Optional[Path] = None, overrides: Optional[list[str]] = None) -> Config:
    raw: dict[str, Any] = {}
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    raw = _expand_env(raw)
    # apply overrides
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set overrides must be key=value: {ov}")
        k, v = ov.split("=", 1)
        _set_nested(raw, k.strip(), _coerce(v.strip()))
    return Config(**raw)


def _set_nested(d: dict, key: str, val: Any) -> None:
    keys = key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = val


def _coerce(raw: str) -> Any:
    low = raw.lower()
    if low in ("true", "yes"): return True
    if low in ("false", "no"): return False
    if low in ("null", "none"): return None
    try:
        if "." in raw or "e" in low: return float(raw)
        return int(raw)
    except ValueError:
        return raw
