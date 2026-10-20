"""Optional local pretrained translation and summarization, never implicit text upload."""
from __future__ import annotations
from functools import lru_cache
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT/".model-cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TRANSLATORS = {("English", "Hindi"): "Helsinki-NLP/opus-mt-en-hi",
               ("Hindi", "English"): "Helsinki-NLP/opus-mt-hi-en"}
SUMMARY_MODEL = "google-t5/t5-small"


@lru_cache(maxsize=3)
def load_model(model_id: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    torch.set_num_threads(2)
    revisions_path = ROOT/"model_revisions.json"
    revision = json.loads(revisions_path.read_text()).get(model_id) if revisions_path.exists() else None
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision)
    model.eval()
    return tokenizer, model


def token_chunks(text, tokenizer, max_tokens=350) -> list[str]:
    """No silent tail truncation. Every input token is assigned to a chunk."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > 3500: raise ValueError("Use text with at most 3,500 model tokens")
    return [tokenizer.decode(tokens[i:i+max_tokens], skip_special_tokens=True)
            for i in range(0,len(tokens),max_tokens)]


def translate(text: str, source="English", target="Hindi") -> dict:
    if not text.strip(): raise ValueError("Extract or enter text first")
    if source == target: return {"text": text, "model": "identity", "seconds": 0., "chunks": 1}
    model_id = TRANSLATORS.get((source,target))
    if not model_id: raise ValueError("Translation currently supports English to Hindi and Hindi to English")
    from torch import inference_mode
    start = time.perf_counter()
    tok, model = load_model(model_id)
    chunks = token_chunks(text, tok)
    outputs = []
    with inference_mode():
        for chunk in chunks:
            batch = tok(chunk, return_tensors="pt", truncation=False)
            result = model.generate(**batch, max_new_tokens=512, num_beams=4, do_sample=False)
            outputs.append(tok.decode(result[0], skip_special_tokens=True))
    return {"text": "\n".join(outputs), "model": model_id, "seconds": time.perf_counter()-start,
            "chunks": len(chunks), "note": "Local pretrained inference. Translation quality is not covered by OCR accuracy metrics."}


def summarize(text: str, language="English") -> dict:
    if language != "English": raise ValueError("Summarization supports English. Translate Hindi text to English first.")
    if len(text.split()) < 30: raise ValueError("Use at least 30 words for summarization")
    from torch import inference_mode
    start = time.perf_counter()
    tok, model = load_model(SUMMARY_MODEL)
    chunks = token_chunks(text,tok)
    outputs = []
    with inference_mode():
        for chunk in chunks:
            batch = tok("summarize: "+chunk, return_tensors="pt", truncation=False)
            result = model.generate(**batch, max_new_tokens=100, num_beams=4,
                                    no_repeat_ngram_size=3, do_sample=False)
            outputs.append(tok.decode(result[0], skip_special_tokens=True))
    return {"text": "\n".join(outputs), "model": SUMMARY_MODEL, "seconds": time.perf_counter()-start,
            "chunks": len(chunks), "note": "Review generated summaries for omissions or unsupported statements."}
