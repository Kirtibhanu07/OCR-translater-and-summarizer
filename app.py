"""Run with: .venv/bin/streamlit run app.py"""
import hashlib
import json
import streamlit as st
from ocr_engine import LANGUAGES, OCRConfig, document_pages, extract_text, installed_languages, default_config
from nlp_tasks import translate, summarize

st.set_page_config(page_title="Document Reader", page_icon="📄", layout="centered")
st.title("Document Reader")
st.write("Extract text from an image or PDF, then translate or summarize it.")
try:
    available = installed_languages()
except (RuntimeError, OSError) as exc:
    st.error(str(exc)); st.stop()

language = st.selectbox("Document language", [k for k,v in LANGUAGES.items() if v in available])
layout = st.selectbox("Page layout", ["Single text block", "Automatic layout"],
                      help="Use automatic layout for pages with multiple columns or separate text regions.")
config = default_config() if layout=="Single text block" else OCRConfig()
uploaded = st.file_uploader("Choose an image or PDF", type=["png","jpg","jpeg","tif","tiff","webp","pdf"])
st.caption("Up to 20 MB and five PDF pages. Text stays on this computer during processing.")

if uploaded is None:
    from pathlib import Path
    sample = Path(__file__).parent/"benchmark_fixtures/eng_05_clean.png"
    if sample.exists() and st.button("Try an English sample"):
        from PIL import Image
        with st.spinner("Reading the sample..."):
            with Image.open(sample) as im: result = extract_text(im.convert("RGB"), "eng", config)
        st.session_state.ocr_text = result["text"]
        st.session_state.ocr_results = [result]
        st.session_state.ocr_text_language = "English"
        st.session_state.pop("translation",None)
        st.session_state.pop("summary",None)

if uploaded is not None:
    data = uploaded.getvalue()
    identity = hashlib.sha256(data+language.encode()+layout.encode()).hexdigest()
    if st.session_state.get("document_id") != identity:
        for key in ["ocr_text","ocr_results","translation","summary"]: st.session_state.pop(key,None)
        st.session_state.document_id = identity
    if st.button("Extract text", type="primary"):
        try:
            with st.spinner("Reading your document..."):
                pages = document_pages(data,uploaded.name)
                results = [extract_text(page, LANGUAGES[language], config) for page in pages]
            st.session_state.ocr_text = "\n\n".join(r["text"] for r in results)
            st.session_state.ocr_results = results
            st.session_state.ocr_text_language = language
            if not st.session_state.ocr_text.strip(): st.warning("No text was found. Try a clearer image or check the language.")
        except Exception as exc:
            st.error(str(exc))

if "ocr_text" in st.session_state:
    def invalidate_generated_text():
        st.session_state.pop("translation",None)
        st.session_state.pop("summary",None)
    text = st.text_area("Extracted text", key="ocr_text", height=260, on_change=invalidate_generated_text)
    elapsed = sum(r["seconds"] for r in st.session_state.ocr_results)
    st.caption(f"Processed {len(st.session_state.ocr_results)} page(s) in {elapsed:.2f} seconds.")
    st.download_button("Download text", text, file_name="extracted_text.txt", mime="text/plain")
    with st.expander("Processing details"):
        st.json({"pages": st.session_state.ocr_results})
        st.caption("Confidence values come from the OCR engine. They are not accuracy scores.")
    st.subheader("Translate or summarize")
    st.caption("Models download on first use. Processing then runs locally. English and Hindi translation and English summarization are available.")
    source = st.selectbox("Text language", list(LANGUAGES), index=list(LANGUAGES).index(st.session_state.get("ocr_text_language",language)))
    target = st.selectbox("Translate to", ["Hindi","English"])
    if st.button("Translate", disabled=source not in ["English","Hindi"]):
        try:
            with st.spinner("Translating locally..."): st.session_state.translation = translate(text,source,target)
        except Exception as exc: st.error("Translation unavailable: "+str(exc))
    if "translation" in st.session_state:
        st.text_area("Translation",st.session_state.translation["text"],height=160)
        st.caption("Translation is experimental. Check meaning, names and numbers against the source.")
    if st.button("Summarize", disabled=source!="English"):
        try:
            with st.spinner("Summarizing locally..."): st.session_state.summary = summarize(text,source)
        except Exception as exc: st.error("Summarization unavailable: "+str(exc))
    if "summary" in st.session_state:
        st.write(st.session_state.summary["text"])
        st.caption("Check the summary against the extracted text before using it.")
