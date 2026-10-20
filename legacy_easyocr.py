import streamlit as st
import easyocr
#from googletrans import Translator
from PIL import Image
import numpy as np
from googletrans import Translator
from google_trans_new import google_translator
import spacy
from heapq import nlargest
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
nlp = spacy.load("en_core_web_sm")

translator = google_translator()


def display_text(bounds):
    context = []
    for x in bounds:
        t = x[1]
        context.append(t)
    context = ' '.join(context)
    return context


def summarizer(text):
    punc = punctuation
    doc = nlp(text)
    tokens = [token.text for token in doc]
    word_freq = {}
    stop_words = list(STOP_WORDS)
    for word in doc:
        if word.text.lower() not in stop_words:
            if word.text.lower() not in punc:
                if word.text not in word_freq.keys():
                    word_freq[word.text] = 1
                else:
                    word_freq[word.text] += 1

    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq

    # Sentence tokenization
    sent_tokens = [sent for sent in doc.sents]

    sent_score = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text.lower() in word_freq.keys():
                if sent not in sent_score.keys():
                    sent_score[sent] = word_freq[word.text.lower()]
                else:
                    sent_score[sent] += word_freq[word.text.lower()]
    summary = nlargest(n=3  , iterable=sent_score, key=sent_score.get)
    final = [word.text for word in summary]
    output = " ".join(final)
    calc = len(output) / len(text) * 100
    record = "Your Summary is " + str(calc) + "% of original text."
    return output, record

st.sidebar.title('Language Selection Menu')
st.sidebar.subheader('Select Language...')
src = st.sidebar.selectbox("From Language", ['English', 'Kannada', 'Hindi', 'Tamil'])

st.sidebar.subheader('Select...')
destination = st.sidebar.selectbox("To Language", ['Hindi', 'Tamil', 'Kannada', 'English'])

helper = {'Hindi': 'hi', 'Kannada': 'kn', 'English': 'en', 'Tamil': 'ta'}
dst = helper[destination]
source = helper[src]

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('AI HACK')
st.subheader('Optical Character Recognition and Summarizer ')
st.text("Summarizer takes only English Language as Input")
st.text('Select source Language from the Sidebar.')

image_file = st.file_uploader("Upload Image", type=['png', 'jpeg', 'JPG'])
text = ""
if image_file is not None:
    st.subheader('Image you Uploaded...')
    st.image(image_file, width=450)

if st.button("Summarize"):

    if image_file is not None:
        img = Image.open(image_file)
        img = np.array(img)

    if src == 'English':
        with st.spinner('Extracting Text from given Image'):
            eng_reader = easyocr.Reader(['en'])
            detected_text = eng_reader.readtext(img)
            st.subheader('Extracted text is ...')
            text = display_text(detected_text)
    summ = summarizer(str(text))
    st.write(text)
    st.write(summ)
    st.balloons()

if st.button("Convert"):

    if image_file is not None:
        img = Image.open(image_file)
        img = np.array(img)

        if src == 'English':
            with st.spinner('Extracting Text from given Image'):
                eng_reader = easyocr.Reader(['en'])
                detected_text = eng_reader.readtext(img)
            st.subheader('Extracted text is ...')
            text = display_text(detected_text)
            st.write(text)

        elif src == 'Hindi':
            with st.spinner('Extracting Text from given Image'):
                hindi_reader = easyocr.Reader(['hi'])
                detected_text = hindi_reader.readtext(img)
            st.subheader('Extracted text is ...')
            text = display_text(detected_text)
            st.write(text)

        elif src == 'Tamil':
            with st.spinner('Extracting Text from given Image'):
                tamil_reader = easyocr.Reader(['ta'])
                detected_text = tamil_reader.readtext(img)
            st.subheader('Extracted text is ...')
            text = display_text(detected_text)
            st.write(text)

        elif src == 'Kannada':
            with st.spinner('Extracting Text from given Image...'):
                kannada_reader = easyocr.Reader(['kn'])
                detected_text = kannada_reader.readtext(img)
            st.subheader('Extracted text is ...')
            text = display_text(detected_text)
            st.write(text)
        st.write('')

        with st.spinner('Translating Text...'):
            result = translator.translate(text, lang_tgt=f'{dst}')

        st.subheader("Translated Text is ...")
        st.write(result)
        st.balloons()

    else:
        st.subheader('Image not found! Please Upload an Image.')
