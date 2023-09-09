import logging
import sys
import os
from langchain.document_loaders import (PyPDFLoader,CSVLoader,TextLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
# from llama_index import GPTVectorStoreIndex
# from llama_index import download_loader
from flask import (Flask, redirect, render_template, request,send_from_directory, url_for)
import json
from flask_cors import CORS

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
os.environ["OPENAI_API_KEY"] = "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"

def saveModelDoc(filepath):    
    loader = TextLoader(filepath)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    data=loader.load()    
    texts = text_splitter.split_documents(data)    
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')    
    Chroma.from_documents(texts, embeddings,persist_directory="./model")

def saveModelPdf(filepath):
    loader = PyPDFLoader(filepath)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    data=loader.load()    
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002') 
    Chroma.from_documents(texts, embeddings,persist_directory="./model")
    

def queryModel(question):        
    db3 = Chroma(persist_directory="./model",embedding_function=OpenAIEmbeddings(model='text-embedding-ada-002'))
    docs = db3.similarity_search(question)
    chain = load_qa_chain(ChatOpenAI(temperature=1,model_name='gpt-3.5-turbo',max_tokens=1000), 
                        chain_type="stuff")    
    response=chain.run(input_documents=docs, question=question)    
    return response

# saveModelDoc('data/indice/serviciosdetalle.txt')
# saveModelPdf('data/tarifario/TARIFARIO-TASA-ABO-21AGO2023.pdf')
# saveModelPdf('data/tarifario/TARIFARIO-ABO-21JUL2023.pdf')

app = Flask(__name__)
CORS(app)
@app.route('/bot-question', methods=['POST'])
def response_question():
    question = request.get_json()
    return queryModel(question['question'])

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)