# CCB Pure Native Draw.io Artifact v42

Source image: `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`

This artifact is a pure native draw.io reconstruction. It does not use raster
overlays, `shape=image`, embedded `data:image`, base64 payloads, or stencil
shapes.

## Metrics

- Overall score: `0.832624`
- Edge F1: `0.7514`
- Edge precision: `0.8325`
- OCR F1: `0.8590`
- OCR precision: `0.9493`
- Changed pixel ratio @30: `0.163858`
- Native purity: `true`

## Search Delta

Baseline was v40 at `0.826588`.

The accepted edits were native text primitive repairs:

- Add `Validated Novel`
- Add `Distillation`
- Increase `Distillation` font size by 1 and shift it slightly
- Add `Fast Feedback`
- Increase `Fast Feedback` font size by 1 and shift it upward by 3 px

The targeted bottom-left and bottom-right title additions were evaluated but
rejected by the metric because they reduced OCR precision.

## Remaining Gaps

The weakest tiles remain bottom-right and bottom-left:

- Bottom-right tile score: `0.755304`, OCR F1 `0.6905`
- Bottom-left tile score: `0.757402`, OCR F1 `0.7333`

The next useful direction is a joint text grouping/layout repair pass for these
tiles, plus native edge routing/icon primitive improvements rather than adding
many duplicate OCR labels.
