import logging
import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from jsonschema import Draft7Validator
# from llama_index import GPTVectorStoreIndex
# from llama_index import download_loader
from flask import (Flask, redirect, render_template, request,send_from_directory, url_for)
import json
from flask_cors import CORS
#from genson import SchemaBuilder

def obtainJSONSchema(json_file_path):
    with open(json_file_path, 'r') as archivo:
        datos_json = json.load(archivo)
    esquema = Draft7Validator(schema={})
    for error in esquema.iter_errors(datos_json):
        print("Error de validación:")
        print(error.message)
        print("Ruta del error:", list(error.path))
    return esquema.schema

def readJsonschema(json_file_path):
    with open(json_file_path, 'r') as archivo:
        datos_json = json.load(archivo)

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
    chain = load_qa_chain(ChatOpenAI(temperature=0.2,model_name='gpt-3.5-turbo',max_tokens=1000), 
                        chain_type="stuff")    
    response=chain.run(input_documents=docs, question=question)
    return response
#saveModel()
app = Flask(__name__)
CORS(app)

@app.route('/bot-question', methods=['POST'])
def response_question():
   question = request.get_json()
   return queryModel(question['question'])

if __name__ == '__main__':
   app.run(host='0.0.0.0',port=5000)