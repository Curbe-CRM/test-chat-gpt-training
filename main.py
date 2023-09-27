import logging
import sys
import os
from flask_jwt_extended import JWTManager,create_access_token,jwt_required, get_jwt_identity
from langchain.document_loaders import (PyPDFLoader,CSVLoader,TextLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
# from llama_index import GPTVectorStoreIndex
# from llama_index import download_loader
from flask import (Flask, redirect, render_template, request,send_from_directory, url_for,jsonify, abort)
import json
from flask_cors import CORS
import openai
from werkzeug.datastructures import ImmutableMultiDict

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ["OPENAI_API_KEY"] = ""

trsted_proxies=('localhost:4200')

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
    chain = load_qa_chain(ChatOpenAI(temperature=0.1,model_name='gpt-3.5-turbo',max_tokens=1000), 
                        chain_type="stuff")
    response=chain.run(input_documents=docs, question=question)
    responseObj={
        'data':{
            'content':response
        }
    }
    return responseObj

def saveFile(request):    
    audio = request.files['file']
    path='audio.mp3';
    audio.save(path)
    return speechText(path)

def speechText(file):
    audio_file= open(file, "rb")
    openai.api_key=os.environ["OPENAI_API_KEY"]
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript['text'])
    return transcript['text']

# saveModelDoc('data/indice/serviciosdetalle.txt')
# saveModelPdf('data/tarifario/TARIFARIO-TASA-ABO-21AGO2023.pdf')
# saveModelPdf('data/tarifario/TARIFARIO-ABO-21JUL2023.pdf')
# saveModelDoc('data/pdfs.txt')

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = ""
jwt = JWTManager(app)
CORS(app)

@app.route('/bot-question', methods=['POST'])
@jwt_required()
def response_question():
    question = request.get_json()
    return queryModel(question['question'])

@app.route('/get-audio', methods=['POST'])
@jwt_required()
def get_audio():
    responseObj={
        'data':{
            'content':saveFile(request)
        }
    } 

@app.route("/token", methods=["POST"])
def create_token():
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    if (username=='initUserTest' and password=='initP455W0RD'):
        access_token = create_access_token(identity='initUser')
        return jsonify({ "token": access_token, "user_id": 'initUser' })
    else:
        abort(403)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)