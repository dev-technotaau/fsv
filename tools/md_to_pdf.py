"""md_to_pdf.py — render PROJECT_WORKFLOW.md to a branded PDF + HTML.

Produces:
  PROJECT_WORKFLOW.html — standalone styled HTML (open in any browser)
  PROJECT_WORKFLOW.pdf  — print-quality PDF via Chrome headless

The MD is converted line-for-line (no content omitted). The only structural
change vs the raw MD is the workflow diagram: the ASCII codeblock is replaced
with a proper SVG flowchart so the visual reads as a polished graphic.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows so emoji/arrow log lines don't crash
# with UnicodeEncodeError under the default cp1252 codepage.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import markdown  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "PROJECT_WORKFLOW.md"
HTML_PATH = ROOT / "PROJECT_WORKFLOW.html"
PDF_PATH = ROOT / "PROJECT_WORKFLOW.pdf"

# ---------------------------------------------------------------------------
# SVG flowchart — replaces the ASCII codeblock in the MD.
# Mirrors the S-curve in the original diagram (1→2→3, drop, 4→5→6, drop, 7).
# ---------------------------------------------------------------------------
SVG_FLOWCHART = """
<svg viewBox="0 0 900 540" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="Fence Stain Simulator workflow diagram"
     class="flow-svg">
  <defs>
    <linearGradient id="g-card" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%"  stop-color="#FFFFFF"/>
      <stop offset="100%" stop-color="#FDF4EC"/>
    </linearGradient>
    <linearGradient id="g-num" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%"  stop-color="#E58939"/>
      <stop offset="100%" stop-color="#8A4A0F"/>
    </linearGradient>
    <linearGradient id="g-final" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%"  stop-color="#14171A"/>
      <stop offset="100%" stop-color="#0B0D0E"/>
    </linearGradient>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#C06B1A"/>
    </marker>
    <filter id="card-shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="3" stdDeviation="6" flood-color="#0B0D0E" flood-opacity="0.10"/>
    </filter>
  </defs>

  <!-- Row 1 -->
  <g filter="url(#card-shadow)">
    <rect x="30"  y="30" width="240" height="110" rx="14" fill="url(#g-card)" stroke="#E5E8EB" stroke-width="1.5"/>
    <rect x="320" y="30" width="240" height="110" rx="14" fill="url(#g-card)" stroke="#E5E8EB" stroke-width="1.5"/>
    <rect x="610" y="30" width="240" height="110" rx="14" fill="url(#g-card)" stroke="#E5E8EB" stroke-width="1.5"/>
  </g>
  <g font-family="'Plus Jakarta Sans', 'Inter', sans-serif">
    <circle cx="65" cy="65" r="18" fill="url(#g-num)"/>
    <text x="65" y="71" text-anchor="middle" font-size="18" font-weight="700" fill="#fff">1</text>
    <text x="95" y="72" font-size="16" font-weight="700" fill="#0B0D0E">Image collection</text>
    <text x="50" y="110" font-size="12.5" fill="#3A4047">3 coordinated scrapes</text>
    <text x="50" y="128" font-size="12.5" fill="#3A4047">~34,000 images, 11+ sources</text>

    <circle cx="355" cy="65" r="18" fill="url(#g-num)"/>
    <text x="355" y="71" text-anchor="middle" font-size="18" font-weight="700" fill="#fff">2</text>
    <text x="385" y="72" font-size="16" font-weight="700" fill="#0B0D0E">Validate &amp; catalog</text>
    <text x="340" y="110" font-size="12.5" fill="#3A4047">Integrity, dedup, PII, license</text>
    <text x="340" y="128" font-size="12.5" fill="#3A4047">→ manifest.jsonl</text>

    <circle cx="645" cy="65" r="18" fill="url(#g-num)"/>
    <text x="645" y="71" text-anchor="middle" font-size="18" font-weight="700" fill="#fff">3</text>
    <text x="675" y="72" font-size="16" font-weight="700" fill="#0B0D0E">Split + Golden set</text>
    <text x="630" y="110" font-size="12.5" fill="#3A4047">70/15/15 stratified</text>
    <text x="630" y="128" font-size="12.5" fill="#3A4047">+ curated QA reference</text>
  </g>

  <!-- Arrows row 1 -->
  <line x1="275" y1="85" x2="315" y2="85" stroke="#C06B1A" stroke-width="2.2" marker-end="url(#arrow)"/>
  <line x1="565" y1="85" x2="605" y2="85" stroke="#C06B1A" stroke-width="2.2" marker-end="url(#arrow)"/>

  <!-- Right-down connector to row 2 -->
  <path d="M 730 145 L 730 200" stroke="#C06B1A" stroke-width="2.2" fill="none" marker-end="url(#arrow)"/>

  <!-- Row 2 (reversed flow) -->
  <g filter="url(#card-shadow)">
    <rect x="610" y="210" width="240" height="110" rx="14" fill="url(#g-card)" stroke="#E5E8EB" stroke-width="1.5"/>
    <rect x="320" y="210" width="240" height="110" rx="14" fill="url(#g-card)" stroke="#E5E8EB" stroke-width="1.5"/>
    <rect x="30"  y="210" width="240" height="110" rx="14" fill="url(#g-card)" stroke="#E5E8EB" stroke-width="1.5"/>
  </g>
  <g font-family="'Plus Jakarta Sans', 'Inter', sans-serif">
    <circle cx="645" cy="245" r="18" fill="url(#g-num)"/>
    <text x="645" y="251" text-anchor="middle" font-size="18" font-weight="700" fill="#fff">4</text>
    <text x="675" y="252" font-size="16" font-weight="700" fill="#0B0D0E">AI labeling</text>
    <text x="630" y="290" font-size="12.5" fill="#3A4047">DINO + SAM 2.1 → auto-QA</text>
    <text x="630" y="308" font-size="12.5" fill="#3A4047">→ SAM 3 manual refine</text>

    <circle cx="355" cy="245" r="18" fill="url(#g-num)"/>
    <text x="355" y="251" text-anchor="middle" font-size="18" font-weight="700" fill="#fff">5</text>
    <text x="385" y="252" font-size="16" font-weight="700" fill="#0B0D0E">Train model</text>
    <text x="340" y="290" font-size="12.5" fill="#3A4047">DINOv2-Small ~31M params</text>
    <text x="340" y="308" font-size="12.5" fill="#3A4047">augment + EMA + final eval</text>

    <circle cx="65" cy="245" r="18" fill="url(#g-num)"/>
    <text x="65" y="251" text-anchor="middle" font-size="18" font-weight="700" fill="#fff">6</text>
    <text x="95" y="252" font-size="16" font-weight="700" fill="#0B0D0E">Deploy to Modal</text>
    <text x="50" y="290" font-size="12.5" fill="#3A4047">ONNX + FastAPI endpoint</text>
    <text x="50" y="308" font-size="12.5" fill="#3A4047">serverless, scale-to-zero</text>
  </g>

  <!-- Arrows row 2 (right→left) -->
  <line x1="605" y1="265" x2="565" y2="265" stroke="#C06B1A" stroke-width="2.2" marker-end="url(#arrow)"/>
  <line x1="315" y1="265" x2="275" y2="265" stroke="#C06B1A" stroke-width="2.2" marker-end="url(#arrow)"/>

  <!-- Left-down connector to row 3 -->
  <path d="M 150 325 L 150 380" stroke="#C06B1A" stroke-width="2.2" fill="none" marker-end="url(#arrow)"/>

  <!-- Row 3: final dark card -->
  <g filter="url(#card-shadow)">
    <rect x="30" y="390" width="820" height="120" rx="16" fill="url(#g-final)" stroke="#1F2326" stroke-width="1.5"/>
  </g>
  <g font-family="'Plus Jakarta Sans', 'Inter', sans-serif">
    <circle cx="65" cy="430" r="22" fill="url(#g-num)"/>
    <text x="65" y="437" text-anchor="middle" font-size="20" font-weight="800" fill="#fff">7</text>
    <text x="105" y="425" font-size="20" font-weight="800" fill="#FFFFFF">Fence Stain Simulator</text>
    <text x="105" y="452" font-size="13.5" fill="rgba(255,255,255,0.78)">Upload → Detect (server) → Color → Download — recolor &amp; UI run locally, no photos stored</text>
    <text x="105" y="478" font-size="12.5" font-style="italic" fill="rgba(255,255,255,0.55)">Privacy-first, zero-infrastructure, cross-platform</text>
  </g>
</svg>
"""

# ---------------------------------------------------------------------------
# CSS — Fence Stain Simulator brand. Print-optimized.
# ---------------------------------------------------------------------------
STYLE_CSS = r"""
:root {
  --brand: #C06B1A;
  --brand-light: #E58939;
  --brand-dark: #8A4A0F;
  --brand-50: #FDF4EC;
  --brand-100: #F8E3CC;
  --ink-900: #0B0D0E;
  --ink-700: #1F2326;
  --ink-600: #3A4047;
  --ink-500: #5B636B;
  --ink-300: #C7CCD2;
  --ink-200: #E5E8EB;
  --ink-100: #F0F2F4;
  --ink-50:  #F7F8FA;
  --paper:   #FFFFFF;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.55;
  color: var(--ink-700);
  background: var(--ink-50);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}
.page {
  max-width: 880px;
  margin: 0 auto;
  padding: 36px 44px 56px;
  background: var(--paper);
}
header.cover {
  background: linear-gradient(135deg, #14171A 0%, #0B0D0E 100%);
  color: #fff;
  padding: 44px 44px 38px;
  margin: -36px -44px 36px;
  border-radius: 0;
  position: relative;
  overflow: hidden;
}
header.cover::before {
  content: '';
  position: absolute;
  top: -100px; right: -80px;
  width: 360px; height: 360px;
  background: radial-gradient(closest-side, rgba(192,107,26,0.42), transparent 70%);
  filter: blur(20px);
  pointer-events: none;
}
.cover-tag {
  display: inline-block;
  font-family: 'JetBrains Mono', 'Cascadia Code', Consolas, monospace;
  font-size: 0.72rem;
  color: rgba(255,255,255,0.65);
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.1);
  padding: 4px 12px;
  border-radius: 999px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 14px;
  position: relative;
  z-index: 1;
}
.cover-title {
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
  font-size: 2.6rem;
  font-weight: 800;
  letter-spacing: -0.025em;
  margin: 0 0 8px;
  color: #fff;
  position: relative;
  z-index: 1;
}
.cover-title .accent {
  background: linear-gradient(120deg, #FFB76A, #C06B1A 60%, #8A4A0F);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.cover-sub {
  color: rgba(255,255,255,0.72);
  font-size: 1rem;
  max-width: 640px;
  margin: 0;
  position: relative;
  z-index: 1;
}

h1, h2, h3 {
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
  color: var(--ink-900);
  letter-spacing: -0.015em;
  font-weight: 700;
  page-break-after: avoid;
}
h1 { display: none; }   /* the title is rendered in the cover banner instead */
h2 {
  font-size: 1.5rem;
  margin: 36px 0 14px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--brand-100);
  page-break-before: auto;
  break-inside: avoid;
}
h2:first-of-type { margin-top: 8px; }
h3 {
  font-size: 1.05rem;
  margin: 22px 0 8px;
  color: var(--brand-dark);
}
p { margin: 0 0 12px; }
ul, ol { margin: 0 0 14px; padding-left: 22px; }
li { margin: 4px 0; }
strong { color: var(--ink-900); }
em { color: var(--ink-700); }

hr {
  border: 0;
  border-top: 1px dashed var(--ink-200);
  margin: 28px 0;
}

a { color: var(--brand-dark); text-decoration: none; border-bottom: 1px solid var(--brand-100); }
a:hover { border-bottom-color: var(--brand); }

code {
  font-family: 'JetBrains Mono', 'Cascadia Code', Consolas, Menlo, monospace;
  font-size: 0.9em;
  background: var(--ink-100);
  padding: 1px 6px;
  border-radius: 4px;
  color: var(--brand-dark);
}
pre {
  background: #0E1012;
  color: #E5E8EB;
  padding: 16px 18px;
  border-radius: 10px;
  overflow-x: auto;
  font-family: 'JetBrains Mono', 'Cascadia Code', Consolas, Menlo, monospace;
  font-size: 0.82rem;
  line-height: 1.5;
  margin: 14px 0 18px;
  border: 1px solid #1F2326;
  page-break-inside: avoid;
}
pre code { background: transparent; color: inherit; padding: 0; }

table {
  width: 100%;
  border-collapse: collapse;
  margin: 14px 0 22px;
  font-size: 0.92rem;
  background: var(--paper);
  border: 1px solid var(--ink-200);
  border-radius: 10px;
  overflow: hidden;
  page-break-inside: avoid;
}
th {
  background: linear-gradient(135deg, var(--brand-50), #fff);
  color: var(--ink-900);
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
  font-weight: 700;
  text-align: left;
  padding: 10px 14px;
  border-bottom: 2px solid var(--brand-100);
  font-size: 0.85rem;
  letter-spacing: 0.01em;
}
td {
  padding: 10px 14px;
  border-top: 1px solid var(--ink-100);
  vertical-align: top;
  color: var(--ink-700);
}
tr:last-child td { border-bottom: none; }
tr:nth-child(even) td { background: rgba(247,248,250,0.5); }

/* The SVG flowchart wrapper */
.flow-svg {
  display: block;
  width: 100%;
  height: auto;
  margin: 18px 0 24px;
  background: linear-gradient(180deg, #FAFBFC 0%, #F7F8FA 100%);
  border: 1px solid var(--ink-200);
  border-radius: 14px;
  padding: 18px 12px;
  page-break-inside: avoid;
}

footer.foot {
  margin: 48px -44px -56px;
  padding: 22px 44px;
  background: var(--ink-50);
  border-top: 1px solid var(--ink-200);
  color: var(--ink-500);
  font-size: 0.85rem;
  text-align: center;
  font-style: italic;
}

/* Print sizing */
@page {
  size: A4;
  margin: 14mm 12mm;
}
@media print {
  body { background: var(--paper); }
  .page { box-shadow: none; padding: 0; max-width: none; }
  header.cover { margin: 0 0 24px; border-radius: 0; }
  footer.foot { margin: 36px 0 0; }
  h2 { page-break-after: avoid; }
  table, pre, .flow-svg { page-break-inside: avoid; }
}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Fence Stain Simulator — Project Workflow</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap">
  <style>{css}</style>
</head>
<body>
  <article class="page">
    <header class="cover">
      <span class="cover-tag">Project workflow · v1</span>
      <h1 class="cover-title">Fence Stain Simulator<br><span class="accent">Project Workflow</span></h1>
      <p class="cover-sub">{intro}</p>
    </header>
    {body}
    <footer class="foot">Fence Stain Simulator — instant AI fence color previews, in any browser.</footer>
  </article>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build() -> int:
    md_text = MD_PATH.read_text(encoding="utf-8")
    if not md_text.strip():
        print(f"ERROR: {MD_PATH} is empty", file=sys.stderr)
        return 2

    # Strip the H1 title (handled by the cover banner)
    md_text = re.sub(r"^# .*\n", "", md_text, count=1)
    # Strip the leading lede paragraph (also surfaced as cover sub-title)
    intro_match = re.search(
        r"A client-friendly walk-through.*?and what each one does\.",
        md_text, flags=re.DOTALL,
    )
    intro_text = (intro_match.group(0).replace("\n", " ")
                  if intro_match else "End-to-end workflow document.")
    if intro_match:
        md_text = md_text.replace(intro_match.group(0), "", 1)

    # Strip the trailing italic line (footer banner already has it)
    md_text = re.sub(r"\*Fence Stain Simulator — .*?\*\s*$", "", md_text).rstrip() + "\n"

    # Replace the ASCII flowchart codeblock with a placeholder
    placeholder = "<!--FLOWCHART_PLACEHOLDER-->"
    flow_re = re.compile(r"```\s*\n[\s\S]*?7\. Fence Stain[\s\S]*?```", re.MULTILINE)
    md_text, n = flow_re.subn(placeholder, md_text, count=1)
    if n == 0:
        print("WARN: did not match flowchart codeblock — SVG will not be inserted.")

    # Convert MD to HTML
    body_html = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "nl2br"],
    )
    body_html = body_html.replace(
        placeholder, f'<div class="flowchart">{SVG_FLOWCHART}</div>'
    )

    full_html = HTML_TEMPLATE.format(
        css=STYLE_CSS, body=body_html, intro=intro_text,
    )
    HTML_PATH.write_text(full_html, encoding="utf-8")
    print(f"[ok] wrote {HTML_PATH}  ({HTML_PATH.stat().st_size/1024:.1f} KB)")

    # PDF via Chrome headless
    chrome = _find_chrome()
    if chrome is None:
        print("WARN: Chrome / Edge not found — PDF not generated. "
              "Open the HTML file and Print → Save as PDF.")
        return 0

    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=10000",
        f"--print-to-pdf={PDF_PATH}",
        HTML_PATH.as_uri(),
    ]
    print(f"[run] {' '.join(cmd[:3])} ... → {PDF_PATH.name}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as e:
        print(f"WARN: chrome PDF run failed: {e.stderr.decode(errors='ignore')[:400]}")
        return 1
    except subprocess.TimeoutExpired:
        print("WARN: chrome timed out after 120s.")
        return 1

    if PDF_PATH.exists():
        print(f"[ok] wrote {PDF_PATH}  ({PDF_PATH.stat().st_size/1024:.1f} KB)")
    return 0


def _find_chrome() -> str | None:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    for bin_ in ("chrome", "google-chrome", "chromium", "msedge"):
        path = shutil.which(bin_)
        if path:
            return path
    return None


if __name__ == "__main__":
    sys.exit(build())
