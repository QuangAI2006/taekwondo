# Data Inclusion and Exclusion Audit

Protocol-level design:

- 140 first-year students.
- 3 kick types: front, roundhouse, axe.
- 2 attempts per kick type.
- Expected collection: 840 videos.

Current active retained pose sequences:

| Split | Front | Roundhouse | Axe | Total |
|---|---:|---:|---:|---:|
| train | 245 | 159 | 244 | 648 |
| val | 38 | 25 | 41 | 104 |
| test/release examples | 5 | 6 | 2 | 13 |
| total | 288 | 190 | 287 | 765 |

Known non-active materials:

- 57 side-kick videos and 57 side-kick sequences are disabled/excluded from the three-class study.
- One front-kick video is under `_excluded_missing_sequence`.
- One cleaning backup exists for a previously misplaced sequence.

The current files do not preserve a complete per-video exclusion log. Therefore, counts for failed pose extraction, unstable tracking, severe occlusion, missing action phases, category mismatch, and other exclusions cannot be reconstructed reliably from the present filesystem alone.

Conclusion: the manuscript can report the auditable retained counts, but should not provide detailed exclusion-reason counts unless the author provides the original cleaning log.

