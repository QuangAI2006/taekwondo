# Expert Agreement Audit

Source workbook: `<DESKTOP>\Sensors_Submission_Package_20260524_081544\source_backups\human_ai_agreement.xlsx`.

N = 30 videos. The workbook contains 30 expert-evaluated samples.

| Metric | Result |
|---|---:|
| Expert score reliability | ICC(A,3) = 0.915, 95% bootstrap CI [0.850, 0.949] |
| Classification accuracy vs. expert majority | 83.3% (25/30) |
| Classification agreement | Cohen's kappa = 0.750 |
| Spearman correlation | rho = 0.627, permutation p = 0.0004, 95% CI [0.317, 0.839] |
| Pearson correlation | r = 0.606, permutation p = 0.0005, 95% CI [0.333, 0.835] |
| Mean absolute error | 0.601 points |
| Mean squared error | 0.514 points^2 |
| Bland-Altman bias | +0.341 points |
| Bland-Altman 95% limits of agreement | [-0.915, 1.598] |
| Samples within limits of agreement | 93.3% |

Per-class MAE by expert-majority class:

| Class | MAE (points) |
|---|---:|
| front | 0.530 |
| roundhouse | 0.733 |
| axe | 0.540 |

Important limitation: because participant identifiers are unavailable, this expert subset cannot currently be verified as participant-independent from the development data. The manuscript should describe it as an expert-annotated subset unless participant-level metadata are later supplied.
