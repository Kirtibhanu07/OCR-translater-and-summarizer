# Tesseract document reader and evaluation

This is a working, AI-assisted extension of the original EasyOCR project. It adds
Tesseract 5 OCR, image and PDF input, explicit language selection, preprocessing,
local MarianMT translation, local T5 summarization, a Streamlit app, and a
source-separated OCR benchmark. The original implementation is preserved in
`legacy_easyocr.py` and its README in `LEGACY_README.md`.

## Supported features

| Feature | Implemented and tested scope |
|---|---|
| OCR | Printed English, Hindi, Tamil, Telugu and Kannada |
| Files | PNG, JPEG, TIFF, WebP and PDF, up to 20 MB and five PDF pages |
| Layout | Single text block or automatic page layout |
| Preprocessing | Optional deskew, scaling, Otsu or adaptive thresholding in the engine |
| Translation | English to Hindi and Hindi to English with pretrained MarianMT |
| Summarization | English with pretrained T5-small |
| Output | Editable extracted text, text download, word boxes and engine confidence |
| Privacy | OCR and model inference run locally. Model files download on first use |

This is a local application. It has not been deployed as a production service.
Tesseract and the Transformer models are pretrained. This work does not claim
training an OCR or language model from scratch.

## Run the application

The current workspace already has the required local environments:

```bash
.venv/bin/streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open `http://127.0.0.1:8501`, choose the document language and layout, then upload a
file or try the bundled English example. Translation is available only for English
and Hindi. Summarization expects at least 30 English words. Review generated
translations and summaries before using them. The smoke translation contains
meaning errors, so execution success must not be represented as quality validation.

## Install on another computer

Use Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-tesseract.lock
```

Install Tesseract 5 plus `eng`, `hin`, `tam`, `tel` and `kan` language packs using
your operating system's package manager. For example, macOS Homebrew supports
`brew install tesseract tesseract-lang`. Debian package names are in `packages.txt`.
Confirm with `tesseract --list-langs`. Set `TESSERACT_CMD` if the binary is outside
your PATH. Do not point the old EasyOCR environment at this application's files.

An exact conda-forge package list for the tested macOS arm64 Tesseract environment
is saved in `tesseract-macos-arm64.lock`. With micromamba installed, it can be
recreated locally with:

```bash
micromamba create -y -p ./.tesseract-env --file tesseract-macos-arm64.lock
```

Optional NLP models can be downloaded in advance:

```bash
.venv/bin/python download_models.py
.venv/bin/python smoke_nlp.py
```

Weights are not included in the project ZIP. The download script records exact
upstream commit hashes in `model_revisions.json`. The runtime uses those revisions.
The translation and summary models require several hundred MB of disk and memory.
Downloaded files are kept in the project-local `.model-cache` directory.

## Measured OCR results

Configuration selection used 30 validation images from six source documents. The
held-out test used 60 images from twelve different source documents. Each source
has clean, rotated, low-contrast, blurred and compressed variants. Inputs are
synthetic printed pages, not independent real-world scans.

| Configuration | Character error rate | Word error rate | Median OCR time |
|---|---:|---:|---:|
| Raw Tesseract, automatic segmentation PSM 3 | 9.16% | 15.85% | 0.083 s |
| Validation-selected single-block PSM 6 | 0.91% | 6.31% | 0.079 s |

The selected change is page segmentation, not image preprocessing. The more
complex preprocessing candidates did not beat raw PSM 6 on validation. The 8.25
percentage-point absolute CER reduction has a paired source-cluster bootstrap
95% interval of 4.43 to 12.04 points. The relative CER reduction is about 90.1% on
this benchmark only. It must not be advertised as a universal accuracy gain.

Tamil is the weakest tested language, with 4.65% CER and 42.31% WER. Those figures
show why one overall score is insufficient. There are only two test source texts
per Indic language, so the language estimates are particularly limited.

Read `benchmark_results/REPORT.md`, `test_by_language.csv`, `test_by_condition.csv`,
and `test_predictions.csv` for the full evidence. `nlp_smoke.json` preserves actual
translation and summary output. It contains no BLEU, ROUGE or semantic-quality
claim.

## Reproduce evaluation

The ZIP includes the 90 benchmark images and exact reference text. This avoids
requiring the original macOS fonts to rerun recognition:

```bash
.venv/bin/python benchmark.py
.venv/bin/python -m pytest -q test_ocr.py
```

To regenerate images on macOS, use `benchmark.py --generate`. Original font names
and hashes are saved in the manifest, while the font binaries are not redistributed.
MuPDF's HarfBuzz shaping handles Indic scripts correctly. A font directory can be
supplied using `--font-dir`.

The frozen protocol is in `benchmark_plan.json`. Selection uses validation CER.
Source documents do not cross the split. The chosen configuration is saved before
test evaluation and drives the app's default. CER uses Unicode NFC code points and
collapsed whitespace, preserving punctuation and case. WER uses whitespace tokens.
Latency includes preprocessing and OCR, but excludes PDF rendering and app/network
overhead. Engine confidence is never reported as recognition accuracy.

## Test coverage

The tests cover edit-distance correctness, Unicode normalization, source split
integrity, invalid language handling, corrupted and oversized files, transparent
images, blank-page deskew, PDF page limits, real OCR execution, PDF rendering,
token-preserving NLP chunking and the Streamlit sample flow. Separate offline NLP
smoke tests execute both translation directions and summarization.

## Sources

- [Tesseract documentation](https://tesseract-ocr.github.io/tessdoc/)
- [Tesseract image-quality guidance](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html)
- [English to Hindi MarianMT](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi)
- [Hindi to English MarianMT](https://huggingface.co/Helsinki-NLP/opus-mt-hi-en)
- [T5-small](https://huggingface.co/google-t5/t5-small)

Completed locally on 20 October 2026. Review and reproduce this assisted extension
before discussing its implementation and results in an interview.
