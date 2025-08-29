from langchain_community.document_loaders import (
    WebBaseLoader, YoutubeLoader, PyPDFLoader, CSVLoader, TextLoader)
import os
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import streamlit as st
from fake_useragent import UserAgent
from time import time


# Define um User-Agent para evitar bloqueio
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

def load_web_content(url):
    document = ''
    for i in range(5):
        try: 
            os.environ["USER_AGENT"] = UserAgent().random
            loader = WebBaseLoader(url)
            list_documents = loader.load()
            document = '\n\n'.join([doc.page_content for doc in list_documents])
            break
            
        except:
            print(f'Erro ao carregar o site. Tentativa {i+1}/5')
            time.sleep(3)
        
        if document == '':
            st.error('Não foi possível carregar o site. Tente outro link.')
            st.stop()
            
    return document

def load_youtube_content(video_id, lang="pt"):
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.list(video_id)
    transcript = transcript_list.find_transcript([lang])
    data = transcript.fetch()  # retorna lista de FetchedTranscriptSnippet    
    documents = '\n\n'.join([snippet.text for snippet in data]) # usa atributo .text
    return documents

def load_csv_content(file_path):
    loader = CSVLoader(file_path=file_path)
    list_documents = loader.load()
    for doc in list_documents:
        doc.page_content = doc.page_content.encode('latin-1').decode('utf-8')  # Converte de latin-1 para utf-8) 
    documents = '\n\n'.join([doc.page_content for doc in list_documents])
    return documents

def load_pdf_content(file_path):
    loader = PyPDFLoader(file_path)
    list_documents = loader.load()
    documents = '\n\n'.join([doc.page_content for doc in list_documents])
    return documents

def load_txt_content(file_path):
    loader = TextLoader(file_path)
    list_documents = loader.load()
    documents = '\n\n'.join([doc.page_content for doc in list_documents])
    return documents

def load_xlsx_content(path):
    df = pd.read_excel(path)
    return df.to_markdown(index=False)

