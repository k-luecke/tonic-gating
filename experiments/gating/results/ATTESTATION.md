# Attestation — Oscillatory Gating vs Learned Router (primary)

**Owner:** Kyle (ZetaZero) / k-luecke  
**Attested_at (PT):** 2026-09-21T19:43:07-05:00  
**Purpose:** External timestamp anchor for the preregistered primary run. Does not alter the result.

## Verdict
**mixed** (PREREG §6): clean ACC and C_act clauses pass; relative shared-drift gracefulness at σ_c = 0.05 rad/cycle fails.

## Content hashes (SHA-256)

| Artifact | SHA-256 |
|----------|---------|
| PREREG.md | c81851b03cf5920c5f34a5fe6ea81411b6cef73012d2fc99561ad155df1fef10 |
| results/PREREG.sha256 (freeze record) | eec159f037179847f1048445bc0523caeb648644d67870e94fa9f7c891b3920c |
| RESULTS.md (incl. exploratory §8 on router drift resilience) | 73cacbcf70ce63ddbb7d862c359fe0352efbb16b3d62228a2a38bbfe360578bf |
| results/comparison_primary/H1_verdict.txt | 3a60c4e9a0c6d4e2311159d141899c35f288407ebfa91528662859043bbf9e67 |
| results/comparison_primary/summary.csv | 91031a32af992d467acb9954c3a23476fe6f9fd0ed91fd6982170e4ac896474f |
| results/comparison_primary/run_meta.json | 814dcd75d8ac3fb7a11cb04149090e9d23667043399e244eb5c5f65e7b2b583b |

## Run clock (from run_meta.json)
- PREREG hash created: 2026-09-21T19:16:41-05:00
- Primary start: 2026-09-21T19:16:46-05:00
- Mode: primary · 200 symbols × 10 seeds

## Drive home
Driftbound Labs / experiments / gating (Google Drive). This gist only anchors hashes and clocks; it is not a code dump.

## Note
RESULTS.md §8 (exploratory) interprets why the no-pilot router held under shared drift (per-symbol digital demod vs stateful gain-window dynamics). That section is interpretation, not an H1 rescue.
