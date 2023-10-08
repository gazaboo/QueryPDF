import streamlit as st
# from dotenv import load_dotenv
from PyPDF2 import PdfReader
import requests
import os
import torch
import pandas as pd
from sentence_transformers.util import semantic_search
from langchain.text_splitter import RecursiveCharacterTextSplitter


def main():
    # load_dotenv()
    st.set_page_config(page_title='Chat with PDF', page_icon=':books:')
    st.header('Query PDFs')

    user_question = st.text_input('What would you like to find in the PDFs ?')

    if user_question:
        if 'dataset_embeddings' in st.session_state:
            answers_sourced = handle_user_input(user_question)
            for answer, source in answers_sourced.items():
                st.write(answer)
                st.markdown(f':orange[Source : {source}]')
                st.divider()
        else:
            st.write('Provide PDFs and click on process first')

    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader(
            'Import here', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing'):
                raw_text, meta_infos = get_pdf_text(pdf_docs)
                st.session_state.meta_infos = meta_infos

                text_chunks = get_text_chunks(raw_text)
                embeddings = query(text_chunks)
                embeddings = pd.DataFrame(embeddings)

                st.session_state.text_chunks = text_chunks
                st.session_state.dataset_embeddings = torch.from_numpy(
                    embeddings.to_numpy()).to(torch.float)


def handle_user_input(input):
    user_query = []
    user_query.append(input)
    query_embedding = query(user_query)
    query_embedding = torch.from_numpy(
        pd.DataFrame(query_embedding).to_numpy()).to(torch.float)

    hits = semantic_search(
        query_embedding, st.session_state.dataset_embeddings, top_k=5)

    ids = [hit['corpus_id'] for hit in hits[0]]
    answers = [st.session_state.text_chunks[id_chunk] for id_chunk in ids]

    sources = []
    for chunk in answers:
        for text, name in st.session_state.meta_infos.items():
            if chunk in text:
                sources.append(name)

    return {answer: source for answer, source in zip(answers, sources)}


def query(texts):
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    TOKEN = st.secrets['HUGGINGFACEHUB_API_TOKEN']
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = requests.post(api_url, headers=headers, json={
                             "inputs": texts, "options": {"wait_for_model": True}})
    return response.json()


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_pdf_text(pdf_docs):
    text = ''
    meta_info = dict()
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text().replace('\n', ' ')
            meta_info[page.extract_text().replace('\n', ' ')] = pdf.name

    return text, meta_info


if __name__ == '__main__':
    main()
