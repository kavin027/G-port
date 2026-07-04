# ACM Template Notes

SoCC 2026 requires the ACM Proceedings Format with:

```tex
\documentclass[sigconf, review, anonymous]{acmart}
```

Official links:

- SoCC 2026 formatting instructions: https://acmsocc.org/2026/papers.html#formatting
- ACM proceedings template page: https://www.acm.org/publications/proceedings-template

The ACM template zip link returned HTTP 403 when fetched from this local
PowerShell session, but TeX Live 2025 already includes `acmart.cls`.  The
working paper in `../main.tex` uses the required class and compiles locally with
`latexmk -pdf main.tex`.
