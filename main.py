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

def saveModelDoc():
    os.environ["OPENAI_API_KEY"] = ""
    loader = TextLoader('data/indice/serviciosdetalle.txt')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    data=loader.load()    
    texts = text_splitter.split_documents(data)
    print(texts[1])
    # embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')    
    # Chroma.from_documents(texts, embeddings,persist_directory="./model")

def saveModel():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    os.environ["OPENAI_API_KEY"] = ""
    loader = PyPDFLoader('../totalidad.pdf')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    data=loader.load()    
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002') 
    docsearch = Chroma.from_documents(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))],persist_directory="./model")    

def queryModel(question):    
    os.environ["OPENAI_API_KEY"] = ""
    db3 = Chroma(persist_directory="./model",embedding_function=OpenAIEmbeddings(model='text-embedding-ada-002'))
    docs = db3.similarity_search(question)
    chain = load_qa_chain(ChatOpenAI(temperature=1,model_name='gpt-3.5-turbo',max_tokens=1000), 
                        chain_type="stuff")    
    response=chain.run(input_documents=docs, question=question)
    # responseObj={
    #     "ref_number_client": "",
    # "ref_number_company": "",
    # "ref_user_name": "",
    # "ref_type": "text",
    # "ref_data": {
    #     "ref_payload": None,
    #     "ref_text": response
    # },
    # "ref_date": "2022-11-21T00:24:36.000Z"
    # }
    return response
saveModelDoc()
# app = Flask(__name__)
# CORS(app)
# @app.route('/bot-question', methods=['POST'])
# def response_question():
#     question = request.get_json()
#     return queryModel(question['question'])

# if __name__ == '__main__':
#     app.run(host='0.0.0.0',port=5000)