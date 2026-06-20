"""build_report.py — render report_build/PROJECT_REPORT.md to a branded PDF + HTML.

Self-contained renderer for the long-form client walkthrough. Based on the
brand styling of tools/md_to_pdf.py but generalized for a 50-60pp document:
robust (no brittle content-specific regexes), auto cover title from the H1,
a generated table of contents, per-section page breaks, and figure/callout
styling. Output:
  report_build/Fence_Stain_Simulator_Technical_Report.html
  report_build/Fence_Stain_Simulator_Technical_Report.pdf  (via Chrome/Edge headless)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import markdown  # type: ignore

HERE = Path(__file__).resolve().parent
MD_PATH = HERE / "PROJECT_REPORT.md"
HTML_PATH = HERE / "Fence_Stain_Simulator_Technical_Report.html"
PDF_PATH = HERE / "Fence_Stain_Simulator_Technical_Report.pdf"

COVER_TITLE = "Fence Stain Simulator"
COVER_SUBTITLE = "End-to-End Technical Walkthrough &amp; Project Report"
COVER_TAG = "June 2026"
COVER_LINK = "https://ninjafencestaining.com/fence-staining-color-simulator/"
COVER_LINK_LABEL = "ninjafencestaining.com/fence-staining-color-simulator"
COVER_LEDE = (
    "A complete, plain-English walk-through of how the Fence Stain Simulator is "
    "built &mdash; from scraping tens of thousands of fence photos, through "
    "AI-assisted labelling and a two-phase deep-learning model, to the live "
    "browser experience your customers use at ninjafencestaining.com &mdash; "
    "with the exact numbers, costs, and decisions behind each step."
)

STYLE_CSS = r"""
:root {
  --brand: #C06B1A; --brand-light: #E58939; --brand-dark: #8A4A0F;
  --brand-50: #FDF4EC; --brand-100: #F8E3CC;
  --ink-900: #0B0D0E; --ink-800:#16191C; --ink-700: #1F2326; --ink-600: #3A4047;
  --ink-500: #5B636B; --ink-400:#8B939B; --ink-300: #C7CCD2; --ink-200: #E5E8EB;
  --ink-100: #F0F2F4; --ink-50: #F7F8FA; --paper: #FFFFFF;
  --ok:#16A34A; --warn:#D97706; --err:#DC2626; --info:#2563EB;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
  font-size: 10.6pt; line-height: 1.55; color: var(--ink-700);
  background: var(--ink-50); -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
}
.page { max-width: 880px; margin: 0 auto; padding: 36px 46px 56px; background: var(--paper); }

/* Cover — matches the live app hero (index4_dinov3.html): dark charcoal + warm orange */
header.cover {
  background: linear-gradient(155deg, #14171A 0%, #0A0C0E 100%); color: #fff;
  padding: 64px 50px 56px; margin: -36px -46px 0; position: relative; overflow: hidden;
  min-height: 96vh; display: flex; flex-direction: column; justify-content: center;
  page-break-after: always; break-after: page;
}
header.cover::before {
  content: ''; position: absolute; top: -120px; right: -80px; width: 480px; height: 480px;
  background: radial-gradient(closest-side, rgba(192,107,26,0.45), transparent 70%);
  filter: blur(40px); pointer-events: none;
}
header.cover::after {
  content: ''; position:absolute; inset:0;
  background:
    linear-gradient(transparent 0, transparent calc(100% - 1px), rgba(255,255,255,0.06) 100%),
    repeating-linear-gradient(90deg, transparent 0, transparent 60px, rgba(255,255,255,0.02) 60px, rgba(255,255,255,0.02) 61px);
  pointer-events:none;
}
.cover-tag {
  display: inline-block; font-family: 'JetBrains Mono', Consolas, monospace; font-size: 0.72rem;
  color: rgba(255,255,255,0.7); background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
  padding: 5px 14px; border-radius: 999px; letter-spacing: 0.08em; text-transform: uppercase;
  margin-bottom: 22px; position: relative; z-index: 1; width: fit-content;
}
.cover-title {
  display: block; font-family: 'Plus Jakarta Sans', 'Inter', sans-serif; font-size: 3.7rem; font-weight: 800;
  letter-spacing: -0.03em; margin: 0 0 8px; color: #ffffff; position: relative; z-index: 1; line-height: 1.0;
}
/* Solid brand-orange accent (no background-clip:text — that renders transparent in some PDF viewers). */
.cover-title .accent { color: #E58939; }
.cover-subtitle { font-family:'Plus Jakarta Sans','Inter',sans-serif; font-size:1.35rem; font-weight:700; color:rgba(255,255,255,0.92); margin:0 0 22px; position:relative; z-index:1; }
.cover-lede { color: rgba(255,255,255,0.74); font-size: 1.02rem; max-width: 660px; margin: 0 0 30px; position: relative; z-index: 1; line-height:1.6; }
.cover-link { position:relative; z-index:1; margin:0 0 10px; font-size:0.95rem; color:rgba(255,255,255,0.9); }
.cover-link .lbl { color:rgba(255,255,255,0.55); font-family:'JetBrains Mono',Consolas,monospace; font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase; display:block; margin-bottom:3px; }
.cover-link a { color:#FFB76A; border-bottom:1px solid rgba(255,183,106,0.4); font-weight:600; }
.cover-foot { margin-top:18px; color:rgba(255,255,255,0.5); font-size:0.8rem; position:relative; z-index:1; font-family:'JetBrains Mono',Consolas,monospace; }

/* markdown H1 is stripped in build(); the cover title is the only h1 and must stay visible — so no global h1{display:none} */
h2 {
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif; color: var(--ink-900); font-weight: 800;
  font-size: 1.7rem; letter-spacing:-0.02em; margin: 0 0 16px; padding: 14px 0 10px;
  border-bottom: 2px solid var(--brand-100); page-break-before: always; break-after: avoid;
}
h2:first-of-type { page-break-before: avoid; }
h3 { font-family: 'Plus Jakarta Sans','Inter',sans-serif; font-size: 1.18rem; font-weight:700; margin: 26px 0 8px; color: var(--ink-900); letter-spacing:-0.01em; break-after: avoid; }
h4 { font-family:'Plus Jakarta Sans','Inter',sans-serif; font-size:1.0rem; font-weight:700; margin:18px 0 6px; color: var(--brand-dark); break-after: avoid; }
p { margin: 0 0 11px; }
ul, ol { margin: 0 0 13px; padding-left: 24px; }
li { margin: 4px 0; }
strong { color: var(--ink-900); }
hr { border:0; border-top:1px dashed var(--ink-200); margin: 26px 0; }
a { color: var(--brand-dark); text-decoration: none; border-bottom: 1px solid var(--brand-100); }

code { font-family: 'JetBrains Mono', Consolas, Menlo, monospace; font-size: 0.86em; background: var(--ink-100); padding: 1px 6px; border-radius: 4px; color: var(--brand-dark); }
pre { background: #0E1012; color: #E5E8EB; padding: 15px 18px; border-radius: 10px; overflow-x: auto; font-family: 'JetBrains Mono', Consolas, Menlo, monospace; font-size: 0.78rem; line-height: 1.5; margin: 13px 0 17px; border: 1px solid #1F2326; page-break-inside: avoid; }
pre code { background: transparent; color: inherit; padding: 0; font-size: inherit; }

table { width: 100%; border-collapse: collapse; margin: 13px 0 20px; font-size: 0.84rem; background: var(--paper); border: 1px solid var(--ink-200); border-radius: 10px; overflow: hidden; page-break-inside: avoid; }
th { background: linear-gradient(135deg, var(--brand-50), #fff); color: var(--ink-900); font-family: 'Plus Jakarta Sans','Inter',sans-serif; font-weight: 700; text-align: left; padding: 9px 13px; border-bottom: 2px solid var(--brand-100); font-size: 0.78rem; }
td { padding: 8px 13px; border-top: 1px solid var(--ink-100); vertical-align: top; color: var(--ink-700); }
tr:nth-child(even) td { background: rgba(247,248,250,0.6); }

blockquote { margin: 14px 0; padding: 12px 18px; background: var(--brand-50); border-left: 4px solid var(--brand); border-radius: 0 10px 10px 0; color: var(--ink-800); }
blockquote p { margin: 0; }

/* Table of contents */
.toc { background: linear-gradient(180deg,#FAFBFC,#F4F6F8); border:1px solid var(--ink-200); border-radius:14px; padding: 20px 26px; margin: 8px 0 0; page-break-after: always; }
.toc-title { font-family:'Plus Jakarta Sans','Inter',sans-serif; font-weight:800; font-size:1.3rem; color:var(--ink-900); margin:0 0 10px; }
.toc ul { list-style:none; padding-left:0; margin:0; }
.toc > ul > li { margin: 7px 0; font-weight:700; color:var(--ink-900); font-size:0.95rem; }
.toc ul ul { padding-left:18px; margin:4px 0; }
.toc ul ul li { font-weight:500; color:var(--ink-600); font-size:0.86rem; margin:2px 0; }
.toc a { border-bottom:none; color:inherit; }

/* Callout note styling via blockquote starting with NOTE/WARN */
@page { size: A4; margin: 16mm 14mm; }
@media print {
  body { background: var(--paper); }
  .page { box-shadow: none; padding: 0; max-width: none; }
  header.cover { margin: 0; min-height: 96vh; page-break-after: always; }
  h2 { page-break-before: always; }
  h2:first-of-type { page-break-before: avoid; }
  table, pre, blockquote { page-break-inside: avoid; }
}
footer.foot { margin: 50px -46px -56px; padding: 22px 46px; background: var(--ink-50); border-top: 1px solid var(--ink-200); color: var(--ink-500); font-size: 0.82rem; text-align: center; font-style: italic; }
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Fence Stain Simulator — Technical Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{css}</style></head>
<body><article class="page">
<header class="cover">
  <span class="cover-tag">{tag}</span>
  <h1 class="cover-title">Fence Stain<br><span class="accent">Simulator</span></h1>
  <p class="cover-subtitle">{subtitle}</p>
  <p class="cover-lede">{lede}</p>
  <p class="cover-link"><span class="lbl">Live tool</span><a href="{link}">{link_label}</a></p>
  <div class="cover-foot">Prepared for Ninja Fence Staining &middot; by TechnoTaau &middot; DINOv3 ViT-L/16 segmentation</div>
</header>
{body}
<footer class="foot">Fence Stain Simulator &mdash; instant AI fence-colour previews, in any browser. &copy; TechnoTaau.</footer>
</article></body></html>
"""


def build() -> int:
    md_text = MD_PATH.read_text(encoding="utf-8")
    if not md_text.strip():
        print(f"ERROR: {MD_PATH} is empty", file=sys.stderr)
        return 2
    # Drop the first H1 (rendered in the cover banner)
    md_text = re.sub(r"^\# .*\n", "", md_text, count=1)

    body_html = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "toc"],
        extension_configs={"toc": {"title": "Contents", "permalink": False}},
    )
    # Wrap the generated TOC (python-markdown emits <div class="toc">) — already styled.

    full_html = HTML_TEMPLATE.format(
        css=STYLE_CSS, body=body_html,
        tag=COVER_TAG, subtitle=COVER_SUBTITLE, lede=COVER_LEDE,
        link=COVER_LINK, link_label=COVER_LINK_LABEL,
    )
    HTML_PATH.write_text(full_html, encoding="utf-8")
    print(f"[ok] wrote {HTML_PATH}  ({HTML_PATH.stat().st_size/1024:.1f} KB)")

    chrome = _find_chrome()
    if chrome is None:
        print("WARN: Chrome/Edge not found — open the HTML and Print → Save as PDF.")
        return 0
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           "--virtual-time-budget=20000", f"--print-to-pdf={PDF_PATH}", HTML_PATH.as_uri()]
    print(f"[run] {Path(chrome).name} --headless → {PDF_PATH.name}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    except subprocess.CalledProcessError as e:
        print(f"WARN: chrome PDF failed: {e.stderr.decode(errors='ignore')[:400]}")
        return 1
    except subprocess.TimeoutExpired:
        print("WARN: chrome timed out.")
        return 1
    if PDF_PATH.exists():
        print(f"[ok] wrote {PDF_PATH}  ({PDF_PATH.stat().st_size/1024:.1f} KB)")
    return 0


def _find_chrome() -> str | None:
    for c in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]:
        if Path(c).exists():
            return c
    for b in ("chrome", "google-chrome", "chromium", "msedge"):
        p = shutil.which(b)
        if p:
            return p
    return None


if __name__ == "__main__":
    sys.exit(build())
