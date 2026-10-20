"""Download allowlisted pretrained models and record exact upstream revisions."""
import json
import os
from pathlib import Path
from nlp_tasks import ROOT, SUMMARY_MODEL, TRANSLATORS
from huggingface_hub import HfApi, snapshot_download

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
models = sorted(set(TRANSLATORS.values()) | {SUMMARY_MODEL})
revision_file = ROOT/"model_revisions.json"
existing = json.loads(revision_file.read_text()) if revision_file.exists() else {}
revisions = {}
for name in models:
    info = HfApi().model_info(name, revision=existing.get(name))
    names = {s.rfilename for s in info.siblings}
    weights = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
    print("Downloading", name, info.sha, flush=True)
    snapshot_download(name, revision=info.sha,
                      allow_patterns=["*.json", "*.spm", "*.txt", "*.model", weights],
                      max_workers=2)
    revisions[name] = info.sha
(ROOT/"model_revisions.json").write_text(json.dumps(revisions,indent=2))
print("Saved exact model revisions",flush=True)
