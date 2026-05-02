"""Download open-access PDFs for the experimental-section bibliography
into REFERENCES/. Prints per-entry success/failure with the resolved URL.

Run from project root:
    python scripts/download_references.py
"""
from __future__ import annotations

import os
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "REFERENCES")

# (key, target_filename, url, source_label, expected_min_kb)
ENTRIES = [
    ("yu2022mappo", "yu2022_mappo.pdf",
     "https://arxiv.org/pdf/2103.01955",
     "arXiv (open)", 100),
    ("stooke2020pid", "stooke2020_pidlagrangian.pdf",
     "https://arxiv.org/pdf/2007.03964",
     "arXiv (open)", 100),
    ("kuba2022happo", "kuba2022_happo.pdf",
     "https://arxiv.org/pdf/2109.11251",
     "arXiv (open)", 100),
    ("gu2023macpo", "gu2023_macpo.pdf",
     "https://arxiv.org/pdf/2110.02793",
     "arXiv (open)", 100),
    ("alshiekh2018shielding", "alshiekh2018_shielding.pdf",
     "https://arxiv.org/pdf/1708.08611",
     "arXiv (open)", 100),
    ("agarwal2021rliable", "agarwal2021_rliable.pdf",
     "https://arxiv.org/pdf/2108.13264",
     "arXiv (open)", 100),
    ("terry2021pettingzoo", "terry2021_pettingzoo.pdf",
     "https://arxiv.org/pdf/2009.14471",
     "arXiv (open)", 100),
    ("towers2024gymnasium", "towers2024_gymnasium.pdf",
     "https://arxiv.org/pdf/2407.17032",
     "arXiv (open)", 100),
    ("mudiyanselage2021ergonomic", "mudiyanselage2021_ergonomic.pdf",
     "https://arxiv.org/pdf/2109.15036",
     "arXiv (open) — Electronics MDPI mirror", 100),
    ("cerqueira2024semg", "cerqueira2024_semg.pdf",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC11678945/pdf/sensors-24-08081.pdf",
     "PMC mirror of MDPI Sensors (open access)", 100),
    # The two we expect to fail (paywalled / 1985 classical):
    ("hubert1985ari", "hubert1985_ari.pdf",
     "https://link.springer.com/content/pdf/10.1007/BF01908075.pdf",
     "Springer J. Classification (LIKELY PAYWALL)", 100),
    ("sun2025perceived", "sun2025_perceived.pdf",
     "https://jneuroengrehab.biomedcentral.com/counter/pdf/10.1186/s12984-025-01787-6.pdf",
     "BMC J. NeuroEng Rehab (BMC is usually open access)", 100),
    ("cataldi2025wearable", "cataldi2025_wearable.pdf",
     "https://www.tandfonline.com/doi/pdf/10.1080/00140139.2025.2486193",
     "Taylor & Francis Ergonomics (LIKELY PAYWALL)", 100),
]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def _download(url: str, path: str, timeout: int = 60) -> tuple[bool, str]:
    """Returns (ok, msg). Saves to `path` on success."""
    try:
        req = Request(url, headers={
            "User-Agent": UA,
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.5",
        })
        with urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "").lower()
            data = resp.read()
            # Light sanity check: PDF magic OR sufficient size + Content-Type hint.
            looks_pdf = data[:4] == b"%PDF" or "pdf" in ctype
            if not looks_pdf:
                return False, f"not a PDF (got Content-Type={ctype}, head={data[:8]!r})"
            with open(path, "wb") as f:
                f.write(data)
            return True, f"{len(data) // 1024} KB"
    except HTTPError as e:
        return False, f"HTTP {e.code} {e.reason}"
    except URLError as e:
        return False, f"network error: {e.reason}"
    except Exception as e:
        return False, f"unexpected: {e!r}"


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    print(f"Target directory: {OUT_DIR}\n")
    for key, fname, url, label, _min_kb in ENTRIES:
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            kb = os.path.getsize(path) // 1024
            print(f"[SKIP] {key:30s} -> {fname} (already exists, {kb} KB)")
            results.append((key, "skip", f"already {kb} KB"))
            continue
        print(f"[ ... ] {key:30s} -> {fname}")
        print(f"        source: {label}")
        print(f"        url:    {url}")
        ok, msg = _download(url, path)
        if ok:
            print(f"        OK: {msg}\n")
            results.append((key, "ok", msg))
        else:
            print(f"        FAIL: {msg}\n")
            results.append((key, "fail", msg))
        # Be polite to publisher servers.
        time.sleep(0.7)

    print("=" * 64)
    print("Summary:")
    ok = sum(1 for _, s, _ in results if s == "ok")
    fail = sum(1 for _, s, _ in results if s == "fail")
    skip = sum(1 for _, s, _ in results if s == "skip")
    print(f"  {ok} downloaded, {fail} failed, {skip} skipped")
    if fail:
        print("\nFailed entries (need manual download or a research-mode fetch):")
        for key, status, msg in results:
            if status == "fail":
                print(f"  - {key}: {msg}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
