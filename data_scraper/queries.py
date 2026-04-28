"""Query generation — ~200 diverse static queries + optional Gemini expansion.

Categories (designed for a multi-class fence model that learns occlusion + distractors):
  A. Wood types + fence styles           (core "fence" positives)
  B. Scene / environment / lighting      (diversity)
  C. Obstacles in front (trees, plants)  (occlusion training data)
  D. Animals / humans near fence         (distractor + context)
  E. Similar-material distractors        (wooden house, pole, shed)
  F. Close-up vs far-distance framing    (scale diversity)
  G. Weather / time-of-day               (lighting diversity)
  H. Multiple fences / partial fences    (structural diversity)
"""
from __future__ import annotations

import itertools
from typing import Optional


# ============================================================
# STATIC QUERY CORPUS
# ============================================================

# A — wood types & styles (heavy weight on cedar per project priority)
WOOD_FENCE_STYLES = [
    "cedar fence",
    "cedar privacy fence",
    "cedar shadowbox fence",
    "cedar picket fence",
    "cedar stockade fence",
    "cedar horizontal slat fence",
    "cedar split rail fence",
    "cedar board on board fence",
    "wooden privacy fence",
    "wooden picket fence",
    "wooden shadowbox fence",
    "wooden horizontal fence",
    "wooden dog-ear fence",
    "wooden scalloped fence",
    "wooden lattice top fence",
    "wooden fence panels",
    "redwood fence",
    "pressure treated wood fence",
    "pine wood fence",
    "bamboo fence",
    "rustic wooden fence",
    "weathered wood fence",
    "new cedar fence installation",
    "freshly stained cedar fence",
    "dark stained cedar fence",
    "natural cedar wood fence",
]

# Non-wood styles (for generalization)
NON_WOOD_STYLES = [
    "vinyl privacy fence",
    "white vinyl fence",
    "chain link fence backyard",
    "wrought iron fence",
    "aluminum fence",
    "composite fence",
    "metal fence panels",
    "barbed wire fence",
    "wire mesh fence",
]

# B — scene / environment
SCENES = [
    "backyard fence landscaping",
    "front yard fence",
    "garden fence with flowers",
    "suburban backyard wooden fence",
    "residential fence with lawn",
    "fence with patio furniture",
    "fence along sidewalk",
    "fence next to driveway",
    "fence with pergola",
    "fence with deck",
    "fence with pool area",
    "fence around garden beds",
    "fence with raised garden",
    "fence with walkway",
    "fence property line",
    "fence bordering green lawn",
]

# C — obstacles in front (occlusion training data — CRITICAL)
OCCLUSIONS = [
    "wooden fence with tree branches in front",
    "fence behind bushes",
    "fence obscured by climbing vines",
    "fence with overhanging tree limbs",
    "fence with shrubs in front",
    "fence behind tall grass",
    "fence covered in ivy",
    "fence with flower bushes",
    "fence with rose bushes in front",
    "fence partially hidden by plants",
    "fence with hedge in front",
    "fence with tree trunk in front",
    "fence with garden vegetables in front",
    "fence with potted plants",
    "fence with bird feeder hanging",
    "fence with hanging planters",
    "fence with hose coiled",
    "fence with garden tools leaning",
    "fence with ladder leaning on it",
    "fence with bicycle parked",
    "fence with trash bin in front",
    "fence with grill nearby",
    "fence in winter with snow on branches",
    "fence with tall weeds in front",
]

# D — animals / humans near fence (context diversity)
HUMANS_ANIMALS = [
    "dog behind wooden fence",
    "dog in front of fence",
    "dog jumping at fence",
    "cat on wooden fence",
    "squirrel on fence post",
    "bird on fence rail",
    "chickens near wooden fence",
    "rabbit by garden fence",
    "person painting wooden fence",
    "person staining cedar fence",
    "person installing fence",
    "kid playing near fence",
    "person gardening near fence",
    "pet in fenced yard",
    "horse behind wooden fence",
]

# E — similar-material distractors (fence must be SEPARATED from these)
DISTRACTORS = [
    "wooden shed next to fence",
    "wooden house siding near fence",
    "wooden pergola with fence",
    "telephone pole next to fence",
    "wooden gazebo in yard with fence",
    "wooden deck with privacy fence",
    "wooden playhouse in fenced yard",
    "wooden trellis next to fence",
    "wooden arbor with fence",
    "log cabin with wooden fence",
    "wooden stairs next to fence",
    "utility pole with fence",
    "wooden bench against fence",
    "wooden raised planter near fence",
    "wooden retaining wall with fence",
]

# F — close-up vs far scale
SCALES = [
    "close-up wooden fence texture",
    "close-up cedar grain",
    "close up picket fence",
    "wooden fence macro detail",
    "aerial view backyard fence",
    "wide angle fence in landscape",
    "long fence perspective",
    "fence down a long property",
    "distant fence across field",
    "fence corner view",
]

# G — weather / time-of-day
CONDITIONS = [
    "wooden fence golden hour",
    "cedar fence sunset light",
    "fence morning dew",
    "fence overcast day",
    "fence in rain",
    "fence after rain wet",
    "fence with frost",
    "fence with snow",
    "fence in fog",
    "fence harsh midday shadow",
    "fence dappled shade",
    "fence dramatic shadows",
    "fence twilight",
    "fence blue hour",
]

# H — multiple fences / partial fences / gates
VARIATIONS = [
    "broken fence panel",
    "rotting wooden fence",
    "fence gate wooden",
    "fence gate open",
    "fence gate closed",
    "two fences meeting at corner",
    "half finished fence construction",
    "old wooden fence",
    "new wooden fence install",
    "fence post only",
    "wooden fence panels stacked",
    "fence around only part of yard",
    "double gate fence",
    "arbor gate with fence",
    "fence with missing board",
]

# Assembled corpus
ALL_STATIC_QUERIES: list[str] = (
    WOOD_FENCE_STYLES
    + NON_WOOD_STYLES
    + SCENES
    + OCCLUSIONS
    + HUMANS_ANIMALS
    + DISTRACTORS
    + SCALES
    + CONDITIONS
    + VARIATIONS
)


# ============================================================
# Query generation
# ============================================================

def build_queries(
    *,
    static: bool = True,
    custom: Optional[list[str]] = None,
    dedup: bool = True,
    gemini_extra: int = 0,
    gemini_api_key: Optional[str] = None,
) -> list[str]:
    queries: list[str] = []
    if static:
        queries.extend(ALL_STATIC_QUERIES)
    if custom:
        queries.extend(custom)
    if gemini_extra > 0 and gemini_api_key:
        queries.extend(_expand_via_gemini(gemini_api_key, gemini_extra))

    if dedup:
        seen = set()
        out = []
        for q in queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(q.strip())
        queries = out
    return queries


_GEMINI_PROMPT = (
    "Generate {n_extra} diverse web search queries for finding photos of "
    "outdoor wooden (especially cedar) fences in residential backyards. "
    "The photos will train a segmentation model, so include queries for: "
    "fences partially occluded by trees/plants/branches, fences near other "
    "wooden structures (houses, sheds, poles), fences with animals/humans "
    "nearby, fences at close-up and far scales, various weather/lighting. "
    "Return ONLY a JSON array of strings, no prose, no code fences."
)
_GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash",
    "gemini-flash-latest",
    "gemini-pro",
]


def _parse_gemini_response(text: str) -> list[str]:
    """Extract the JSON array of queries from Gemini's text response."""
    import json
    text = (text or "").strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    arr = json.loads(text)
    return [str(q).strip() for q in arr if isinstance(q, str) and q.strip()]


def _read_project_id() -> str:
    """Extract project_id from the service-account JSON pointed to by
    GOOGLE_APPLICATION_CREDENTIALS. Empty string if unavailable."""
    import os, json
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip('"')
    if not cred_path or not os.path.exists(cred_path):
        return ""
    try:
        with open(cred_path, "r", encoding="utf-8") as f:
            return str(json.load(f).get("project_id", "") or "")
    except Exception:
        return ""


def _run_gemini(client, prompt: str, label: str, log_extra: dict) -> list[str]:
    """Try each candidate model through the given google-genai Client.
    Returns the first successful result (list of queries) or [] if all failed."""
    from google.genai import types
    from .logger import get_logger
    log = get_logger("queries")

    cfg = types.GenerateContentConfig(temperature=0.9)
    last_err = ""
    for name in _GEMINI_MODELS:
        try:
            resp = client.models.generate_content(
                model=name, contents=prompt, config=cfg,
            )
            queries = _parse_gemini_response(getattr(resp, "text", "") or "")
            if queries:
                log.info(f"gemini_expansion_ok_{label}",
                         model=name, n=len(queries), **log_extra)
                return queries
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            log.debug(f"gemini_model_failed_{label}", model=name, error=last_err)
            continue
    log.warn(f"gemini_expansion_failed_{label}",
             tried=_GEMINI_MODELS, last_error=last_err)
    return []


def _expand_via_vertex(n_extra: int) -> list[str]:
    """Call Gemini via google-genai SDK in Vertex AI mode. Billing goes through
    Google Cloud (uses the $300 free-trial credits). Auth via the service
    account in GOOGLE_APPLICATION_CREDENTIALS."""
    from .logger import get_logger
    log = get_logger("queries")

    project_id = _read_project_id()
    if not project_id:
        log.debug("vertex_no_credentials",
                  hint="set GOOGLE_APPLICATION_CREDENTIALS to service-account JSON")
        return []

    try:
        from google import genai
    except ImportError:
        log.debug("genai_sdk_missing", hint="pip install google-genai")
        return []

    try:
        client = genai.Client(vertexai=True, project=project_id, location="us-central1")
    except Exception as e:
        log.warn("vertex_init_failed", error=str(e)[:120])
        return []

    return _run_gemini(client, _GEMINI_PROMPT.format(n_extra=n_extra),
                       label="vertex", log_extra={"project": project_id})


def _expand_via_ai_studio(api_key: str, n_extra: int) -> list[str]:
    """Fallback: Gemini via google-genai SDK in AI Studio mode. Uses AI Studio
    billing (separate from Google Cloud credits)."""
    from .logger import get_logger
    log = get_logger("queries")

    try:
        from google import genai
    except ImportError:
        return []
    if not api_key:
        return []
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        log.warn("ai_studio_init_failed", error=str(e)[:120])
        return []

    return _run_gemini(client, _GEMINI_PROMPT.format(n_extra=n_extra),
                       label="aistudio", log_extra={})


def _expand_via_gemini(api_key: str, n_extra: int) -> list[str]:
    """Ask Gemini to generate more diverse queries.

    Prefers Vertex AI (billed through Google Cloud, uses $300 credits). Falls
    back to AI Studio only if Vertex isn't configured/available, so behaviour
    degrades gracefully rather than breaking.
    """
    queries = _expand_via_vertex(n_extra)
    if queries:
        return queries
    return _expand_via_ai_studio(api_key, n_extra)


def round_robin(queries: list[str]):
    """Generator that yields queries in a round-robin cycle forever.
    Useful for coordinator loops that want to distribute queries evenly."""
    for q in itertools.cycle(queries):
        yield q
