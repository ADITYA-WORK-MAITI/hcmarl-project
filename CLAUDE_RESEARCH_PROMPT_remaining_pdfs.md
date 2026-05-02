# Claude Research Mode prompt — fetch 2 remaining HC-MARL reference PDFs

## Status before running this prompt

11 of 13 experimental-section PDFs are already in `REFERENCES/` after
`scripts/download_references.py` ran successfully. The remaining 2 are
paywalled / classical-era and need a more thorough open-access hunt
than my single-shot `urllib` script could do. Paste the prompt below
into Claude Research Mode (or a similar deep-search agent) verbatim.

---

## ===== PASTE BELOW INTO RESEARCH MODE =====

I need direct, open-access PDF download URLs for two academic papers. I
will NOT use Sci-Hub, LibGen, or any unauthorised mirror. I want
**legitimate open-access channels only**: author's personal/lab/
university repository, an institutional repository (e.g., UIUC IDEALS,
DSpace, eScholarship), an OA preprint server (arXiv, SSRN, OSF, HAL,
PsyArXiv, PubMed Central, EuropePMC), or a publisher-hosted free
version. For each paper, I want the FINAL deliverable to be a direct
URL where curling that URL returns `application/pdf` content, OR an
authoritative statement that no such legitimate open-access version
exists.

For each paper, please:
1. Search the author's personal homepage and Google Scholar profile
   for an "Open Access" or "PDF" link.
2. Search OA preprint servers (Google Scholar's "All N versions" link
   is the fastest enumeration).
3. Check institutional repositories of the authors' affiliations.
4. Check ResearchGate / Academia.edu only if author-uploaded (these
   often have legitimate author-deposited copies).
5. Check the publisher's "Open Access Companion Article" or
   "Author Accepted Manuscript" links.
6. Verify the URL by attempting an HTTP GET and confirming
   Content-Type is `application/pdf` and the first 4 bytes are `%PDF`.

If multiple OA versions exist, prefer the latest peer-reviewed version
(VoR > AAM > preprint). If only a preprint version exists, that is
acceptable but please note it explicitly.

If NO legitimate OA version exists, say so directly. Do not link
Sci-Hub. Do not link the publisher's paywalled landing page as if it
were the PDF.

### Paper 1 — Hubert & Arabie 1985

  - Authors: Lawrence Hubert, Phipps Arabie
  - Title: "Comparing Partitions"
  - Journal: Journal of Classification, Vol. 2, Issue 1, Pages 193–218
  - Year: 1985
  - DOI: 10.1007/BF01908075
  - Publisher: Springer
  - Why I need it: foundational reference for the Adjusted Rand Index
    (ARI), used in our paper's MMICRL synthetic-K=3 cluster-recovery
    validation (we report ARI=1.0).

  Notes:
  - Hubert was at the University of Illinois Urbana-Champaign psych
    dept; UIUC has IDEALS as institutional OA repository — worth
    checking there.
  - Arabie was at Rutgers / NJIT; check NJIT DSpace.
  - The paper has 30,000+ citations; statistical-classification
    textbooks often quote whole sections, so a public open-access copy
    likely exists somewhere.
  - Acceptable preprint substitute: any author-deposited version of
    the same content with the same equation numbering. If only the
    Springer paywall exists, say so and I will cite without local PDF
    (TMLR accepts classical references without PDF deposit).

### Paper 2 — Cataldi et al. 2025

  - Title: "Wearable sensors for classification of load-handling tasks
    with machine learning algorithms in occupational safety and
    health: a systematic literature review"
  - Journal: Ergonomics (Taylor & Francis)
  - Year: 2025 (online April 17 2025)
  - DOI: 10.1080/00140139.2025.2486193
  - Publisher: Taylor & Francis (paywalled landing page returns 403
    when curled directly).

  Notes:
  - Author list was NOT cleanly extractable from the publisher page
    (T&F blocks anonymised WebFetch). Please ALSO return the
    confirmed full author list as a side product.
  - The paper is a systematic literature review of 851 studies (15
    included) on wearable + ML for load-handling.
  - Author affiliations are likely Italian universities (the surname
    suggests an Italian primary author); check the institutional
    repositories of likely affiliations (POLITO, UniBO, UniRoma1,
    PoliMi all have OA mandates).
  - Taylor & Francis allows authors to deposit AAMs after a 12–18
    month embargo, so a 2025 paper may not yet have an authorised
    AAM available. If so, say so directly.

### Deliverable format

For each paper, return:
```
PAPER:    <short cite>
STATUS:   FOUND | NOT_FOUND
URL:      <direct PDF URL if FOUND, else explanation>
SOURCE:   <author repo | OA preprint | institutional | publisher>
CONFIRMED: yes/no (HTTP HEAD returns application/pdf, first 4 bytes %PDF)
NOTES:    <anything I should know>
```

If FOUND, also return a one-line `curl` command I can run locally to
deposit the PDF at `REFERENCES/<firstauthor><year>_<keyword>.pdf`.

If NOT_FOUND, explicitly say "no legitimate open-access version
located" so I can decide whether to (a) cite without local PDF, (b)
request institutional access, or (c) drop the citation.

Do NOT recommend Sci-Hub. Do NOT recommend LibGen. Do NOT recommend
Anna's Archive. If those are the only routes, return NOT_FOUND.

## ===== END RESEARCH MODE PROMPT =====

---

## After research mode returns

For each `FOUND` entry, run the returned `curl` command at the project
root to deposit the PDF. For each `NOT_FOUND`, decide whether to:
- Cite without local PDF (the math doc already does this for [25]
  Khalil and [26] Rohmert; TMLR accepts this for classical or
  paywalled-only references), or
- Request the PDF through your institution's library e-resources,
  or
- Drop the citation if it's not load-bearing for the paper.

Hubert & Arabie 1985 is the ARI reference; if no PDF lands, citing
without PDF is the right call (it's the canonical reference and TMLR
reviewers will accept it). Cataldi 2025 is a related-work breadth
citation; if no PDF lands, the paper is not load-bearing and you can
drop it without weakening the contribution.
