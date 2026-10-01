# Replacement Map

## Global Changes

- Replace Transformer-as-final-system wording with LSTM-UNI as the selected recognition backbone.
- Replace any claim of subject-independent evaluation with sample-level validation unless participant metadata are recovered.
- Replace mobile deployment language with desktop-CPU prototype using smartphone-captured videos.
- Replace biomechanical power terminology with kinematic vigor index or impact-speed proxy.

## Abstract

- Update dataset count to 765 retained pose sequences from 840 planned videos.
- Insert the LSTM-UNI result as the main recognition result.
- Include expert agreement only as preliminary evidence.
- Remove state-of-the-art, expert-equivalent, and on-device deployment claims.

## Methods

- Add the leakage audit and participant-ID limitation.
- Add exact 13-keypoint mapping, 52-dimensional feature construction, and 96-frame sequence length.
- State MediaPipe PoseLandmarker configuration from code.
- Clarify that visibility threshold handling differs across scripts unless harmonized.
- Describe LSTM, GCN, and Transformer as compared backbones.
- Describe the scoring rules from `scripts/scoring_api.py`.

## Results

- Replace old recognition tables with Tables 4-6 from `outputs/manuscript_revisions/tables/`.
- Add Table 7 runtime results.
- Add Table 8 expert agreement results.
- Use the LSTM-UNI confusion matrix as the final selected model figure.

## Discussion

- Explain why LSTM-UNI is selected over Transformer under the current data.
- State that the Transformer phase-aligned sweep result is a combined alignment plus inference effect.
- Add limitations on participant leakage, monocular 2D sensing, occlusion, camera view, desktop CPU runtime, and small expert subset.

## Conclusion

- Avoid claims of proven teaching improvement, expert-equivalent scoring, real-time mobile deployment, or state-of-the-art classification.

## Figures and Tables

- Table 1: replace with acquisition protocol table.
- Table 2: replace with dataset inclusion and current split summary.
- Table 3: replace with model architecture/training table.
- Table 4: replace with recognition backbone comparison.
- Table 5: replace with temporal ablation.
- Table 6: replace with class-wise final model performance.
- Table 7: add runtime table.
- Table 8: add human-AI agreement table.
- Figure A: use row-normalized confusion matrix for the selected LSTM-UNI model.
- Figure B: use Bland-Altman plot from `outputs/expert_agreement/`.
- Figure C: optional workflow diagram from `outputs/figures/experimental_workflow.*`.
