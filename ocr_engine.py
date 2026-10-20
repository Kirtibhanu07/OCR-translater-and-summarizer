"""Local Tesseract OCR, explicit language selection and bounded document input."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
LANGUAGES = {"English": "eng", "Hindi": "hin", "Tamil": "tam", "Telugu": "tel", "Kannada": "kan"}
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_PAGES = 5
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


@dataclass(frozen=True)
class OCRConfig:
    name: str = "raw_psm3"
    preprocessing: str = "raw"
    psm: int = 3


CONFIGS = [OCRConfig(), OCRConfig("raw_psm6", "raw", 6),
           OCRConfig("enhanced_psm3", "enhanced", 3),
           OCRConfig("enhanced_psm6", "enhanced", 6),
           OCRConfig("adaptive_psm6", "adaptive", 6)]


def default_config() -> OCRConfig:
    path = ROOT/"benchmark_results/selection.json"
    if path.exists():
        import json
        return OCRConfig(**json.loads(path.read_text())["selected_config"])
    return CONFIGS[0]


def tesseract_path() -> str:
    choices = [os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"),
               str(ROOT/".tesseract-env/bin/tesseract"),
               str(ROOT/".mamba/envs/.tesseract-env/bin/tesseract")]
    for path in choices:
        if path and Path(path).is_file(): return path
    raise RuntimeError("Tesseract is missing. Install Tesseract 5 and the required language packs, or set TESSERACT_CMD.")


def tess_env() -> dict:
    env = os.environ.copy()
    env["OMP_THREAD_LIMIT"] = "1"
    return env


def installed_languages() -> set[str]:
    result = subprocess.run([tesseract_path(), "--list-langs"], capture_output=True,
                            text=True, timeout=10, check=True, env=tess_env())
    return {line.strip() for line in result.stdout.splitlines()[1:] if line.strip()}


def validate_language(language: str) -> None:
    parts = language.split("+")
    if not parts or any(p not in LANGUAGES.values() for p in parts):
        raise ValueError("Choose a supported OCR language")
    missing = set(parts) - installed_languages()
    if missing: raise ValueError(f"Missing Tesseract language packs: {', '.join(sorted(missing))}")


def open_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_BYTES: raise ValueError("Upload an image or PDF no larger than 20 MB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > MAX_PIXELS: raise ValueError("Image exceeds 20 million pixels")
            im.load()
            image = ImageOps.exif_transpose(im).convert("RGBA")
            white = Image.new("RGBA", image.size, "white")
            white.alpha_composite(image)
            return white.convert("RGB")
    except (Image.UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded image cannot be decoded safely") from exc


def document_pages(data: bytes, name: str) -> list[Image.Image]:
    if not data or len(data) > MAX_BYTES: raise ValueError("Upload a document no larger than 20 MB")
    if Path(name).suffix.lower() != ".pdf": return [open_image(data)]
    import pymupdf
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass: raise ValueError("Unlock the PDF before uploading")
            if not 1 <= doc.page_count <= MAX_PAGES: raise ValueError("Use a PDF with one to five pages")
            out = []
            for page in doc:
                scale = 200/72
                if page.rect.width * page.rect.height * scale**2 > MAX_PIXELS:
                    raise ValueError("A PDF page exceeds the rendering limit")
                pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                out.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
            return out
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ValueError("The PDF cannot be opened") from exc


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Choose a small corrective angle from image projections, without reference text."""
    scale = min(1., 900/gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, mask = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if (mask > 0).sum() < 40: return gray, 0.
    h, w = mask.shape
    def score(angle):
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(mask, matrix, (w,h), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rotated.sum(axis=1)))
    angles = np.arange(-5, 5.01, .5)
    scores = [score(a) for a in angles]
    best = float(angles[int(np.argmax(scores))])
    if max(scores) < score(0)*1.05: best = 0.
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w/2, h/2), best, 1)
    return cv2.warpAffine(gray, matrix, (w,h), flags=cv2.INTER_CUBIC, borderValue=255), best


def preprocess(image: Image.Image, mode: str) -> tuple[Image.Image, dict]:
    if image.width * image.height > MAX_PIXELS: raise ValueError("Image exceeds processing limit")
    if mode == "raw": return image.convert("RGB"), {"deskew_degrees": 0., "scale": 1.}
    if mode not in {"enhanced", "adaptive"}: raise ValueError("Unknown preprocessing mode")
    gray = np.asarray(image.convert("L"))
    factor = max(1., min(2., 1600/gray.shape[1]))
    if gray.size * factor**2 > MAX_PIXELS: factor = 1.
    if factor > 1.01: gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    gray, angle = deskew(gray)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    if mode == "adaptive":
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)
    else:
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(gray), {"deskew_degrees": angle, "scale": factor}


def extract_text(image: Image.Image, language="eng", config: OCRConfig | None = None) -> dict:
    cfg = config or default_config()
    if cfg.psm not in {3,6}: raise ValueError("Supported page segmentation modes are 3 and 6")
    if image.width < 10 or image.height < 10: raise ValueError("Image is too small")
    validate_language(language)
    start = time.perf_counter()
    prepared, details = preprocess(image, cfg.preprocessing)
    with tempfile.TemporaryDirectory(prefix="ocr-tesseract-") as tmp:
        source, prefix = Path(tmp)/"page.png", Path(tmp)/"result"
        prepared.save(source)
        command = [tesseract_path(), str(source), str(prefix), "-l", language, "--oem", "1",
                   "--psm", str(cfg.psm), "--dpi", "300", "txt", "tsv"]
        try:
            subprocess.run(command, capture_output=True, text=True, timeout=45, check=True, env=tess_env())
        except subprocess.TimeoutExpired as exc: raise RuntimeError("OCR exceeded the 45-second page limit") from exc
        except subprocess.CalledProcessError as exc: raise RuntimeError("Tesseract could not process this page: "+exc.stderr[:300]) from exc
        text = prefix.with_suffix(".txt").read_text(encoding="utf-8").strip()
        words = []
        with prefix.with_suffix(".tsv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if row.get("level") == "5" and row.get("text", "").strip():
                    words.append({"text": row["text"], "confidence": float(row["conf"]),
                                  **{k: int(row[k]) for k in ["left","top","width","height"]}})
    elapsed = time.perf_counter()-start
    conf = float(np.mean([w["confidence"] for w in words])) if words else None
    return {"text": text, "language": language, "seconds": elapsed, "word_count": len(words),
            "mean_engine_confidence": conf, "words": words, "config": asdict(cfg),
            "preprocessing": details, "note": "Engine confidence is not measured recognition accuracy."}
