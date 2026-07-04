# SoCC 2026 Paper Workspace

Target venue: ACM Symposium on Cloud Computing (SoCC 2026).

Official submission facts checked on 2026-05-25:

- Website: https://acmsocc.org/2026/
- CFP: https://acmsocc.org/2026/papers.html
- Submission site: https://socc26.hotcrp.com
- Second-round abstract deadline: July 7, 2026 AoE.
- Second-round full-paper deadline: July 14, 2026 AoE.
- Full research papers: 12 pages plus unlimited references.
- Review mode for research papers: dual anonymous.
- Required format: ACM Proceedings Format, 9pt.
- Required LaTeX class: `\documentclass[sigconf, review, anonymous]{acmart}`.
- ACM template page: https://www.acm.org/publications/proceedings-template

Build:

```powershell
latexmk -pdf main.tex
```

Artifact repository:

- Anonymous review artifact: packaged with the submission, without repository
  metadata or private endpoints.
- Camera-ready public repository: `<GITHUB_URL_TO_BE_FILLED_AFTER_ACCEPTANCE>`.
  Do not fill this placeholder in the dual-anonymous submission.

The current draft is a SoCC-oriented paper around code-aware runtime scheduling
for sparse flexible coded learning.  Online guarded scheduling is implemented
in the TCP worker-service runtime and validated in the direct K3s matrix; replay
experiments remain as guard ablations.  The current compiled PDF is intended to
fit within the 12-page research-paper limit; the latest checked build is 11
pages including references.

Figure and table reproduction commands are maintained from the repository root:

```powershell
make reproduce-paper-assets
make compile-paper
```

The full artifact command map is in
`docs/socc_artifact_reproduction.md`.

Before submission, run these final checks:

- Verify the generated PDF stays within the 12-page research-paper limit.
- Rebuild the K3s and guard summary tables from the archived diagnostics.
- Rebuild the Algorithm 3 prediction-accuracy and threshold-sensitivity tables
  with `make reproduce-figure3` or `analyze_guard_prediction.py`.
- Check that the dual-anonymous version removes repository names,
  acknowledgments, self-identifying citations, private IPs, hostnames, raw SSH
  logs, and local user paths.
- Keep the theory language scoped to surrogate rationale and runtime
  correctness, not a full minimal-subset probability bound.
