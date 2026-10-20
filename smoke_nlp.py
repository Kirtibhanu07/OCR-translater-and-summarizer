"""Run local pretrained NLP end to end. This verifies execution, not quality."""
import json
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path
from PIL import Image
from ocr_engine import ROOT, extract_text
from nlp_tasks import translate, summarize

ocr_pages=[]
for sample in ["eng_03_clean.png","eng_05_clean.png"]:
    with Image.open(ROOT/"benchmark_fixtures"/sample) as image:
        ocr_pages.append(extract_text(image,"eng"))
source="\n\n".join(p["text"] for p in ocr_pages)
print("OCR ready",flush=True)
forward=translate(source,"English","Hindi")
print("English to Hindi ready",flush=True)
backward=translate(forward["text"],"Hindi","English")
print("Hindi to English ready",flush=True)
summary=summarize(source,"English")
print("Summary ready",flush=True)
assert all(r["text"].strip() for r in [forward,backward,summary])
assert any("\u0900" <= c <= "\u097f" for c in forward["text"])
out={"source":source,"ocr":ocr_pages,"english_to_hindi":forward,"hindi_to_english":backward,
     "english_summary":summary,"verified":"All three pretrained-model features executed locally and returned nonempty output. No translation or summary quality score is claimed."}
(ROOT/"benchmark_results/nlp_smoke.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({"translation":forward["text"],"summary":summary["text"]},ensure_ascii=False),flush=True)
