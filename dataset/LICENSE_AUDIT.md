# Dataset License Audit

_Generated: 2026-04-17T09:01:28+00:00_  
_Manifest: `dataset\manifest.jsonl` (SHA-256 prefix: `d132ccd394e1…`)_

## Executive summary

- **Total images**: 33,423
- **SAFE for commercial use**: 12,367 (37.0%)
- **SAFE with required attribution**: 3,519 (10.5%)
- **RISKY (do not use commercially without clearance)**: 17,537 (52.5%)
- **UNKNOWN sources**: 0

## Recommended production subset

For the deployed client model, train on the **15,886 SAFE + SAFE_ATTR rows only**. 
This excludes 17,537 RISKY rows whose copyright is unverified.

Run with `--emit-safe` to generate `dataset/manifest_safe.jsonl` containing only the legally-clean subset.

## Per-source breakdown

| Source | Count | Tier | Rationale | License URL |
|--------|-------|------|-----------|-------------|
| `pexels` | 6,915 | 🟢 **SAFE** | Pexels License: free for commercial use, no attribution required | [link](https://www.pexels.com/license/) |
| `pw_google` | 5,726 | 🔴 **RISKY** | Scraped via Google Images — images belong to third-party sources. Copyright unverified; commercial use not cleared. | [link](https://policies.google.com/terms) |
| `pw_bing` | 4,675 | 🔴 **RISKY** | Scraped via Bing Images — same as Google. Third-party copyright. | [link](https://www.microsoft.com/en-us/legal/terms-of-use) |
| `pw_houzz` | 4,247 | 🔴 **RISKY** | Houzz photos copyright by original companies / photographers. TOS prohibits unauthorized use. | [link](https://www.houzz.com/terms) |
| `wikimedia` | 3,519 | 🟡 **SAFE_ATTR** | CC-BY / CC-BY-SA / PD variants — attribution required per file | [link](https://commons.wikimedia.org/wiki/Commons:Licensing) |
| `unsplash` | 2,814 | 🟢 **SAFE** | Unsplash License: free commercial use (excludes 'Unsplash+' tier) | [link](https://unsplash.com/license) |
| `pixabay` | 2,638 | 🟢 **SAFE** | Pixabay Content License: free for commercial use, no attribution | [link](https://pixabay.com/service/license-summary/) |
| `pw_pinterest` | 1,965 | 🔴 **RISKY** | Pinterest pins retain original uploader copyright. Pinterest TOS prohibits commercial scraping. | [link](https://policy.pinterest.com/en/terms-of-service) |
| `company_sites` | 924 | 🔴 **RISKY** | Direct scrapes from fence/staining company sites. Copyright held by each company. | [link](Varies per company) |

## Attribution plan (for SAFE_ATTR sources)

Each image from a `SAFE_ATTR` source carries its original attribution metadata in the `extra.license`, `extra.owner`, or `extra.photographer` field of the manifest row.

At deployment time, generate an `ATTRIBUTION.md` by iterating over the training manifest and listing unique (author, license) pairs for all SAFE_ATTR images used.

## Risks & legal notes

- **Web-scraped sources** (pw_google, pw_bing, pw_pinterest, pw_houzz, company_sites) DO NOT clear copyright. Use only for internal R&D or clear per-image before deployment.
- **Fair use / research exemptions** (US) may apply for model training only, not deployment — consult client's legal counsel for specifics.
- **EU AI Act (2024)**: for high-risk systems, training data provenance may be auditable. Maintain this report with the model deliverable.
- **GDPR**: if any images contain identifiable EU individuals, run `tools/scan_pii.py` and address before deployment.

---

_This report is a starting point; final legal sign-off must come from the client's counsel._