"""Generate source-separated multilingual fixtures, select on validation, score test once."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import html
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import time
import unicodedata

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import pymupdf
from ocr_engine import ROOT, CONFIGS, OCRConfig, extract_text, tesseract_path

os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".mplconfig"))
FONTS = Path("/System/Library/Fonts/Supplemental")
FONT_NAMES = {"eng": "Arial.ttf", "hin": "Devanagari Sangam MN.ttc", "tam": "Tamil Sangam MN.ttc",
              "tel": "Telugu Sangam MN.ttc", "kan": "Kannada Sangam MN.ttc"}

# Authored synthetic fixtures, deliberately unrelated to private resume documents.
CORPUS = {
"eng": [
    ("validation", "The analytics team checks each report before delivery. Clear tables and accurate labels help readers understand the results. Data quality is part of the daily workflow."),
    ("validation", "Annual rainfall changes from year to year. Researchers compare past observations with a simple baseline before selecting a forecasting method. The report explains the limitations."),
    ("test", "The regional office received twenty applications on Monday. Each application was reviewed for missing information. The final list will be shared after the verification process is complete."),
    ("test", "A public library offers books, newspapers and digital learning resources. Visitors can use quiet study rooms during opening hours. Staff members help students find reliable sources for their assignments."),
    ("test", "The project dashboard records delivery dates and quality checks. Analysts review unusual values before sending the monthly report. A short explanation accompanies each chart to prevent misunderstanding."),
    ("test", "Small businesses track sales and inventory every week. Comparing actual demand with the forecast helps reduce excess orders. Managers should review both forecast errors and customer feedback before changing the plan.")],
"hin": [
    ("validation", "आज मौसम साफ है। विद्यालय में विद्यार्थियों ने पुस्तकें पढ़ीं। शिक्षक ने सभी प्रश्नों के उत्तर दिए।"),
    ("test", "नगर के पुस्तकालय में नई पुस्तकें आई हैं। पाठक सुबह से शाम तक अध्ययन कर सकते हैं। कर्मचारी जानकारी प्राप्त करने में सहायता करते हैं।"),
    ("test", "टीम ने वर्षा के आंकड़ों का विश्लेषण किया। प्रत्येक रिपोर्ट की जांच की गई। परिणाम समझाने के लिए सरल चित्र बनाए गए।")],
"tam": [
    ("validation", "இன்று வானிலை தெளிவாக உள்ளது. மாணவர்கள் பள்ளியில் புத்தகங்களைப் படித்தனர். ஆசிரியர் கேள்விகளுக்கு பதில் அளித்தார்."),
    ("test", "நகர நூலகத்தில் புதிய புத்தகங்கள் உள்ளன. வாசகர்கள் அமைதியான அறையில் படிக்கலாம். ஊழியர்கள் தகவல்களைத் தேட உதவுகிறார்கள்."),
    ("test", "குழு மழை தரவுகளை ஆய்வு செய்தது. ஒவ்வொரு அறிக்கையும் சரிபார்க்கப்பட்டது. முடிவுகளை விளக்க எளிய வரைபடங்கள் பயன்படுத்தப்பட்டன.")],
"tel": [
    ("validation", "ఈ రోజు వాతావరణం స్పష్టంగా ఉంది. విద్యార్థులు పాఠశాలలో పుస్తకాలు చదివారు. ఉపాధ్యాయుడు ప్రశ్నలకు సమాధానాలు ఇచ్చారు."),
    ("test", "నగర గ్రంథాలయంలో కొత్త పుస్తకాలు ఉన్నాయి. పాఠకులు ప్రశాంతమైన గదిలో చదువుకోవచ్చు. సిబ్బంది సమాచారం కనుగొనడంలో సహాయం చేస్తారు."),
    ("test", "బృందం వర్షపాతం సమాచారాన్ని విశ్లేషించింది. ప్రతి నివేదికను తనిఖీ చేశారు. ఫలితాలను వివరించడానికి సరళమైన చిత్రాలు ఉపయోగించారు.")],
"kan": [
    ("validation", "ಇಂದು ಹವಾಮಾನ ಸ್ಪಷ್ಟವಾಗಿದೆ. ವಿದ್ಯಾರ್ಥಿಗಳು ಶಾಲೆಯಲ್ಲಿ ಪುಸ್ತಕಗಳನ್ನು ಓದಿದರು. ಶಿಕ್ಷಕರು ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರ ನೀಡಿದರು."),
    ("test", "ನಗರ ಗ್ರಂಥಾಲಯದಲ್ಲಿ ಹೊಸ ಪುಸ್ತಕಗಳು ಇವೆ. ಓದುಗರು ಶಾಂತವಾದ ಕೋಣೆಯಲ್ಲಿ ಅಧ್ಯಯನ ಮಾಡಬಹುದು. ಸಿಬ್ಬಂದಿ ಮಾಹಿತಿ ಹುಡುಕಲು ಸಹಾಯ ಮಾಡುತ್ತಾರೆ."),
    ("test", "ತಂಡವು ಮಳೆಯ ಮಾಹಿತಿಯನ್ನು ವಿಶ್ಲೇಷಿಸಿತು. ಪ್ರತಿ ವರದಿಯನ್ನು ಪರಿಶೀಲಿಸಲಾಯಿತು. ಫಲಿತಾಂಶಗಳನ್ನು ವಿವರಿಸಲು ಸರಳವಾದ ಚಿತ್ರಗಳನ್ನು ಬಳಸಲಾಯಿತು.")]
}


def normalize(text):
    return " ".join(unicodedata.normalize("NFC",text).split())


def edit_distance(a, b):
    if len(a) < len(b): a,b = b,a
    previous = list(range(len(b)+1))
    for i, x in enumerate(a,1):
        current = [i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def errors(reference, prediction):
    ref,hyp = normalize(reference),normalize(prediction)
    if not ref: raise ValueError("Reference must contain text")
    ce,we = edit_distance(ref,hyp), edit_distance(ref.split(),hyp.split())
    return {"char_errors":ce,"reference_chars":len(ref),"word_errors":we,"reference_words":len(ref.split()),
            "CER":ce/len(ref),"WER":we/len(ref.split()),"exact_match":ref==hyp}


def render_text(text,lang,font_dir=FONTS):
    filename = FONT_NAMES[lang]
    if not (font_dir/filename).exists(): raise ValueError(f"Font missing: {font_dir/filename}. Use saved fixtures or provide --font-dir.")
    doc=pymupdf.open();page=doc.new_page(width=580,height=240)
    css=f'@font-face {{font-family: bench; src: url("{filename}");}} * {{font-family: bench; font-size: 17px; line-height: 1.45;}}'
    spare,scale=page.insert_htmlbox(pymupdf.Rect(22,22,558,218),html.escape(text),css=css,archive=pymupdf.Archive(str(font_dir)))
    if scale < .99 or spare < 0: raise ValueError("Fixture text does not fit without shrinking")
    pix=page.get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False)
    image=Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
    doc.close()
    return image


def degrade(image, condition, rng):
    if condition=="clean": return image
    if condition=="rotated": return image.rotate(4, resample=Image.Resampling.BICUBIC,expand=True,fillcolor="white")
    if condition=="low_contrast":
        base=np.asarray(image.convert("L"),dtype=float)
        # Text and page have low separation, plus deterministic scanner-like noise.
        low=200+(base/255)*30+rng.normal(0,2.5,base.shape)
        return Image.fromarray(np.uint8(np.clip(low,0,255))).convert("RGB")
    if condition=="blurred": return image.filter(ImageFilter.GaussianBlur(1.2))
    if condition=="compressed":
        reduced=image.resize((image.width//2,image.height//2),Image.Resampling.LANCZOS)
        buffer=io.BytesIO();reduced.save(buffer,format="JPEG",quality=30)
        buffer.seek(0)
        with Image.open(buffer) as decoded: return decoded.convert("RGB")
    raise ValueError(condition)


def generate_fixtures(folder, font_dir=FONTS):
    folder.mkdir(parents=True,exist_ok=True)
    plan=json.loads((ROOT/"benchmark_plan.json").read_text());rng=np.random.default_rng(plan["seed"])
    rows=[]
    for lang,records in CORPUS.items():
        font_path=font_dir/FONT_NAMES[lang]
        font_sha=hashlib.sha256(font_path.read_bytes()).hexdigest()
        for i,(split,text) in enumerate(records):
            document_id=f"{lang}_{i:02d}"
            base=render_text(text,lang,font_dir)
            for condition in plan["conditions"]:
                image=degrade(base,condition,rng)
                name=f"{document_id}_{condition}.png";image.save(folder/name)
                rows.append({"image":name,"document_id":document_id,"split":split,"language":lang,
                             "condition":condition,"reference":text,"font":FONT_NAMES[lang],"font_sha256":font_sha,
                             "image_sha256":hashlib.sha256((folder/name).read_bytes()).hexdigest()})
    (folder/"manifest.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    return rows


def run_samples(samples,configs,folder):
    out=[]
    for cfg in configs:
        print("Running",cfg.name,"on",len(samples),"fixtures",flush=True)
        for index,s in enumerate(samples):
            path=folder/s["image"]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=s["image_sha256"]: raise ValueError("Fixture changed after generation")
            with Image.open(path) as im: image=im.convert("RGB")
            result=extract_text(image,s["language"],cfg)
            out.append({**s,"config":cfg.name,"prediction":result["text"],"OCR_seconds":result["seconds"],
                        **errors(s["reference"],result["text"])})
            if (index+1)%10==0: print(" ",index+1,"done",flush=True)
    return pd.DataFrame(out)


def aggregate(frame,by=("config",)):
    out=[]
    for keys,g in frame.groupby(list(by),sort=True):
        if not isinstance(keys,tuple): keys=(keys,)
        out.append({**dict(zip(by,keys)),"images":len(g),"source_documents":g.document_id.nunique(),
                    "micro_CER":g.char_errors.sum()/g.reference_chars.sum(),
                    "micro_WER":g.word_errors.sum()/g.reference_words.sum(),"macro_CER":g.CER.mean(),
                    "exact_document_match":g.exact_match.mean(),
                    "median_OCR_seconds":g.OCR_seconds.median(),"p95_OCR_seconds":g.OCR_seconds.quantile(.95)})
    return pd.DataFrame(out)


def cluster_bootstrap(frame,selected,baseline="raw_psm3",n=10000):
    base=frame[frame.config==baseline].groupby("document_id")[["char_errors","reference_chars"]].sum().sort_index()
    pick=frame[frame.config==selected].groupby("document_id")[["char_errors","reference_chars"]].sum().reindex(base.index)
    if pick.isna().any().any(): raise ValueError("Unpaired benchmark results")
    rng=np.random.default_rng(42);idx=rng.integers(0,len(base),size=(n,len(base)))
    delta=(base.char_errors.to_numpy()[idx].sum(axis=1)-pick.char_errors.to_numpy()[idx].sum(axis=1))/base.reference_chars.to_numpy()[idx].sum(axis=1)
    observed=base.char_errors.sum()/base.reference_chars.sum()-pick.char_errors.sum()/pick.reference_chars.sum()
    return {"source_document_clusters":len(base),"absolute_CER_reduction_percentage_points":float(observed*100),
            "95pct_cluster_bootstrap_interval_percentage_points":(np.quantile(delta,[.025,.975])*100).tolist(),
            "resamples":n,"seed":42,"note":"Synthetic source-document clusters. Does not establish real-world OCR accuracy."}


def report(out,validation,test,selection,inference,per_language):
    s=selection["selected_config"]["name"];chosen=test[test.config==s].iloc[0]
    table="| Pipeline | CER | WER | Median seconds |\n|---|---:|---:|---:|\n"
    for _,r in test.iterrows(): table+=f"| {r.config} | {r.micro_CER:.2%} | {r.micro_WER:.2%} | {r.median_OCR_seconds:.3f} |\n"
    langs="| Language | Pipeline | CER | WER |\n|---|---|---:|---:|\n"
    for _,r in per_language.iterrows(): langs+=f"| {r.language} | {r.config} | {r.micro_CER:.2%} | {r.micro_WER:.2%} |\n"
    ci=inference['95pct_cluster_bootstrap_interval_percentage_points']
    content=f"""# Tesseract OCR evaluation

The validation-selected pipeline is **{s}**. It has **{chosen.micro_CER:.2%} character
error rate** and **{chosen.micro_WER:.2%} word error rate** on {int(chosen.images)} synthetic
test images. Median OCR pipeline latency is {chosen.median_OCR_seconds:.3f} seconds
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

{table}
## Language breakdown

{langs}
## Paired uncertainty

Absolute CER reduction versus the raw reference is
{inference['absolute_CER_reduction_percentage_points']:.3f} percentage points, with
a 95% paired source-document cluster bootstrap interval of [{ci[0]:.3f}, {ci[1]:.3f}].
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
"""
    (out/"REPORT.md").write_text(content)
    evidence=(f"Implemented multilingual Tesseract OCR with image preprocessing and PDF support. "
              f"Benchmarked five configurations using source-separated validation and 60 synthetic test images across five languages, "
              f"measuring {chosen.micro_CER:.2%} CER, {chosen.micro_WER:.2%} WER and {chosen.median_OCR_seconds:.2f}-second median OCR latency. "
              "These metrics apply only to the synthetic benchmark.")
    (out/"resume_evidence.txt").write_text(evidence+"\n")


def plot_results(out,per_language):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data=per_language.pivot(index="language",columns="config",values="micro_CER")*100
    ax=data.plot.bar(figsize=(9,5),rot=0,color=["#17658b","#d2813a"])
    ax.set(title="Synthetic printed-document OCR test",xlabel="Known input language",ylabel="Character error rate (%)")
    ax.spines[['top','right']].set_visible(False)
    ax.figure.tight_layout();ax.figure.savefig(out/"ocr_comparison.png",dpi=160);plt.close(ax.figure)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--generate",action="store_true")
    parser.add_argument("--font-dir",type=Path,default=FONTS);args=parser.parse_args()
    folder=ROOT/"benchmark_fixtures";out=ROOT/"benchmark_results";out.mkdir(exist_ok=True)
    samples=generate_fixtures(folder,args.font_dir) if args.generate else json.loads((folder/"manifest.json").read_text())
    val=[s for s in samples if s['split']=='validation'];test=[s for s in samples if s['split']=='test']
    if set(s['document_id'] for s in val)&set(s['document_id'] for s in test): raise ValueError("Source leakage")
    vp=run_samples(val,CONFIGS,folder);vm=aggregate(vp).sort_values(["micro_CER","config"])
    name=vm.iloc[0]['config'];cfg=next(c for c in CONFIGS if c.name==name)
    selection={"selected_config":asdict(cfg),"criterion":"validation_micro_CER", "validation_images":len(val),"test_images":len(test)}
    # Lock the configuration before accessing test outcomes.
    (out/"selection.json").write_text(json.dumps(selection,indent=2))
    vp.to_csv(out/"validation_predictions.csv",index=False);vm.to_csv(out/"validation_metrics.csv",index=False)
    print("Locked selection:",name,flush=True)
    cfgs=[CONFIGS[0]] if name==CONFIGS[0].name else [CONFIGS[0],cfg]
    tp=run_samples(test,cfgs,folder);tm=aggregate(tp);langs=aggregate(tp,("language","config"));conditions=aggregate(tp,("condition","config"))
    tp.to_csv(out/"test_predictions.csv",index=False);tm.to_csv(out/"test_metrics.csv",index=False)
    langs.to_csv(out/"test_by_language.csv",index=False);conditions.to_csv(out/"test_by_condition.csv",index=False)
    inference=cluster_bootstrap(tp,name);(out/"inference.json").write_text(json.dumps(inference,indent=2))
    import subprocess
    version=subprocess.run([tesseract_path(),"--version"],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    env={"python":platform.python_version(),"platform":platform.platform(),"tesseract":version,
         "packages":{p:importlib.metadata.version(p) for p in ['pillow','numpy','opencv-python-headless','pymupdf']},
         "benchmark_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "plan_sha256":hashlib.sha256((ROOT/"benchmark_plan.json").read_bytes()).hexdigest(),
         "engine_sha256":hashlib.sha256((ROOT/"ocr_engine.py").read_bytes()).hexdigest()}
    (out/"environment.json").write_text(json.dumps(env,indent=2))
    report(out,vm,tm,selection,inference,langs);plot_results(out,langs)
    print(tm.to_string(index=False),flush=True)


if __name__=="__main__": main()
