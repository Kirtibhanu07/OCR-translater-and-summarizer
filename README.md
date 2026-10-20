# Multilingual Tesseract Document Reader

Working image and PDF OCR with local translation, summarization and reproducible evaluation.

See [the project guide](PROJECT_GUIDE.md) for installation, app usage, benchmark results, limitations and testing.

- [OCR evaluation report](benchmark_results/REPORT.md)
- [Per-language metrics](benchmark_results/test_by_language.csv)
- [Actual NLP smoke-test output](benchmark_results/nlp_smoke.json)

The previous EasyOCR application is preserved in `legacy_easyocr.py`. The new entry point is `app.py`, with `main.py` kept as a compatibility wrapper.
