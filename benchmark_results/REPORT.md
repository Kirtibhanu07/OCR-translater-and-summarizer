# Tesseract OCR evaluation

The validation-selected pipeline is **raw_psm6**. It has **0.91% character
error rate** and **6.31% word error rate** on 60 synthetic
test images. Median OCR pipeline latency is 0.079 seconds
per image on the recorded machine. Lower error is better. Error rates are not
probabilities and can exceed 100% when insertions are numerous.

## Scope and experimental design

This is a small, controlled synthetic benchmark of printed English, Hindi, Tamil,
Telugu and Kannada. It is not a real scanned-document test and must never be
described as general recognition accuracy. Each input language is supplied to the
engine. Font rendering uses MuPDF's complex-script shaping, not unshaped Unicode
text drawn with a Latin-only layout engine.

Six source texts are assigned to validation and twelve different texts to test
before generating five image conditions. This gives 30 validation images and 60
test images. No source text crosses the split. Clean, rotated, low-contrast,
blurred and compressed variants are evaluated. Five fixed configurations are
compared on validation CER, then the selected configuration and raw PSM 3 reference
are evaluated on the held-out source documents. The selected configuration is
saved before test evaluation, and is the application's default.

If the raw reference wins validation, it remains the default. We do not force an
enhancement to claim an improvement. Document errors and latency distributions are
saved so poor language or degradation performance cannot be hidden by averages.

## Test results

| Pipeline | CER | WER | Median seconds |
|---|---:|---:|---:|
| raw_psm3 | 9.16% | 15.85% | 0.083 |
| raw_psm6 | 0.91% | 6.31% | 0.079 |

## Language breakdown

| Language | Pipeline | CER | WER |
|---|---|---:|---:|
| eng | raw_psm3 | 6.63% | 10.00% |
| eng | raw_psm6 | 0.05% | 0.18% |
| hin | raw_psm3 | 1.07% | 0.85% |
| hin | raw_psm6 | 0.25% | 0.43% |
| kan | raw_psm3 | 11.00% | 23.70% |
| kan | raw_psm6 | 1.33% | 11.85% |
| tam | raw_psm3 | 18.84% | 53.85% |
| tam | raw_psm6 | 4.65% | 42.31% |
| tel | raw_psm3 | 14.25% | 22.22% |
| tel | raw_psm6 | 0.32% | 2.22% |

## Paired uncertainty

Absolute CER reduction versus the raw reference is
8.254 percentage points, with
a 95% paired source-document cluster bootstrap interval of [4.429, 12.043].
All five degraded variants stay together during resampling. There are only twelve
independent test source texts. A confidence interval that includes zero does not
establish a clear improvement.

## Reproducibility

Normalization uses Unicode NFC and collapsed whitespace only, preserving case
and punctuation. CER counts code points, including combining characters and
spaces, rather than grapheme clusters. WER uses whitespace-delimited tokens.
Micro error rates divide total edits by total reference length. Latency includes
image preprocessing and OCR subprocess work, excludes PDF decoding and fixture
rendering, and is not a production load test.

See `benchmark_plan.json`, the image manifest and SHA-256 hashes, `selection.json`,
the environment manifest, per-image predictions, and saved language/condition
breakdowns. Run `.venv/bin/python benchmark.py` to evaluate the saved fixtures.

## Limits

No fine-tuning, handwriting evaluation, production deployment or real-world
generalization claim. Fonts and source texts are limited. Translation and
summarization are separate local pretrained-model features and are not measured by
these OCR error rates. Their smoke tests verify execution only, not quality.
