import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

import ocr_engine as engine
from benchmark import errors, normalize, edit_distance, cluster_bootstrap
from nlp_tasks import translate, summarize, token_chunks


def test_exact_text_has_zero_errors():
    r=errors("Clear report 2026.", "Clear report 2026.")
    assert r["CER"]==r["WER"]==0 and r["exact_match"]


def test_insertions_are_not_clipped_to_fake_accuracy():
    r=errors("a", "a b c d")
    assert r["CER"]>1 and r["WER"]>1


def test_known_edit_distances():
    assert edit_distance("kitten","sitting")==3
    assert errors("one two", "one too")["word_errors"]==1


def test_unicode_nfc_whitespace_normalization_preserves_case():
    assert normalize(" cafe\u0301\n  report ")=="café report"
    assert errors("Report", "report")["CER"]>0


def test_invalid_language_rejected_before_subprocess():
    with pytest.raises(ValueError): engine.validate_language("eng; touch x")
    with pytest.raises(ValueError): engine.validate_language("unknown")


def test_bad_image_rejected():
    with pytest.raises(ValueError): engine.open_image(b"not an image")
    with pytest.raises(ValueError): engine.open_image(b"")


def test_oversized_upload_rejected(monkeypatch):
    monkeypatch.setattr(engine,"MAX_BYTES",5)
    with pytest.raises(ValueError): engine.open_image(b"123456")


def test_transparency_composited_on_white():
    im=Image.new("RGBA",(20,20),(0,0,0,0));buf=io.BytesIO();im.save(buf,format="PNG")
    out=engine.open_image(buf.getvalue())
    assert out.getpixel((0,0))==(255,255,255)


def test_blank_deskew_does_not_rotate():
    image=np.full((50,100),255,dtype=np.uint8)
    result,angle=engine.deskew(image)
    assert angle==0
    np.testing.assert_array_equal(result,image)


def test_preprocessing_rejects_unknown_mode():
    with pytest.raises(ValueError): engine.preprocess(Image.new("RGB",(20,20)),"invalid")


def test_pdf_too_many_pages_rejected():
    import pymupdf
    d=pymupdf.open()
    for _ in range(6): d.new_page()
    data=d.tobytes();d.close()
    with pytest.raises(ValueError,match="five pages"): engine.document_pages(data,"test.pdf")


def test_nlp_rejects_unsupported_language_and_short_summary():
    with pytest.raises(ValueError): translate("hello","Tamil","English")
    with pytest.raises(ValueError): summarize("short")
    with pytest.raises(ValueError): summarize("word "*40,"Hindi")


def test_translation_identity_does_not_download_model():
    assert translate("hello","English","English")["text"]=="hello"


def test_chunking_retains_every_token_and_enforces_limit():
    class Tokenizer:
        def encode(self,text,add_special_tokens=False): return list(range(int(text)))
        def decode(self,tokens,skip_special_tokens=True): return ",".join(map(str,tokens))
    chunks=token_chunks("800",Tokenizer(),350)
    flattened=[int(n) for c in chunks for n in c.split(",")]
    assert flattened==list(range(800))
    with pytest.raises(ValueError): token_chunks("3501",Tokenizer())


def require_engine():
    try: engine.tesseract_path()
    except RuntimeError: pytest.skip("Install Tesseract to run integration tests")


def test_actual_tesseract_reads_a_known_page():
    require_engine()
    # Use saved shaped fixture to avoid depending on a platform-specific font.
    sample=engine.ROOT/"benchmark_fixtures/eng_00_clean.png"
    if not sample.exists(): pytest.skip("Generate benchmark fixtures first")
    with Image.open(sample) as image: result=engine.extract_text(image,"eng",engine.CONFIGS[0])
    assert "analytics" in result["text"].lower()
    assert result["word_count"]>15
    assert result["seconds"]>0
    assert all(w["width"]>0 for w in result["words"])


def test_real_pdf_page_rendering():
    import pymupdf
    d=pymupdf.open();p=d.new_page();p.insert_text((50,70),"A clear report.")
    pages=engine.document_pages(d.tobytes(),"test.pdf");d.close()
    assert len(pages)==1 and pages[0].width>1000


def test_validation_test_source_separation_and_saved_metrics():
    folder=engine.ROOT/"benchmark_fixtures"
    if not (folder/"manifest.json").exists(): pytest.skip("Generate fixtures first")
    records=json.loads((folder/"manifest.json").read_text())
    val={r['document_id'] for r in records if r['split']=='validation'}
    test={r['document_id'] for r in records if r['split']=='test'}
    assert not val&test and len(val)==6 and len(test)==12
    assert len(records)==90
    assert {r['language'] for r in records}==set(engine.LANGUAGES.values())


def test_streamlit_initial_screen_and_sample():
    require_engine()
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(engine.ROOT/"app.py")).run(timeout=30)
    assert not app.exception
    samples=[b for b in app.button if b.label=="Try an English sample"]
    if samples:
        samples[0].click().run(timeout=30)
        assert not app.exception
        assert "businesses" in app.text_area[0].value.lower()
