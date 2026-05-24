# CCB Pure Native Draw.io Artifact v43

Source image: `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`

This artifact remains pure native draw.io. It does not use raster overlays,
`shape=image`, embedded `data:image`, base64 payloads, or stencil shapes.

## Metrics

- Overall score: `0.832640`
- Edge F1: `0.7514`
- Edge precision: `0.8326`
- OCR F1: `0.8590`
- OCR precision: `0.9493`
- Semantic OCR F1: `0.8581`
- Semantic OCR precision: `0.9489`
- Changed pixel ratio @30: `0.163859`
- Native purity: `true`

## Accepted Change

v43 applies a low-risk semantic title repair:

- `Al-Enabled` -> `AI-Enabled`
- `Al-Enhanced` -> `AI-Enhanced`
- `Al-Driven` -> `AI-Driven`

The change is tiny in raw metrics but corrects a systematic OCR confusion in
figure captions while preserving visual layout.

## Evaluation Change

The evaluator now records a semantic-normalized OCR sub-metric for common OCR
confusions such as `Al`/`AI`, `Metrocs`/`Metrics`, and `Drand`/`Brand`. The
ranking score uses the better of raw and semantic OCR terms, preventing the
search loop from preferring visibly wrong text solely because the source OCR
misread a noisy label.

## Remaining Gaps

The visual bottleneck is still not the main captions. The weak areas remain:

- bottom-left: agent titles, local labels, icon fidelity, and overlapping text
- bottom-right: benchmarking block, legend wording/layout, chart details, and
  small routing/icon primitives

The next pass should use region-level scene graph repair instead of broad
residual line dumping or deletion of real labels.
