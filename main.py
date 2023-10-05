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
from flask import (Flask, redirect, render_template, request,send_from_directory, url_for,jsonify, abort)
import json
from flask_cors import CORS
import openai
from werkzeug.datastructures import ImmutableMultiDict
import psycopg2

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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
    return response

def saveFileAudio(request):
    audio = request.files['file']
    path='audio.mp3';
    audio.save(path)
    return speechText(path)

def speechText(file):
    audio_file= open(file, "rb")
    openai.api_key=os.environ["OPENAI_API_KEY"]
    transcript = openai.Audio.transcribe("whisper-1", audio_file)    
    return transcript['text']

def consultar_tabla(tabla, columnas="*", condicion=None,multiple=False):    
    try:
        conn = psycopg2.connect(
            database=os.environ['CONN_DB'],
            user=os.environ['CONN_USER'],
            password=os.environ['CONN_PASWD'],
            host=os.environ['CONN_HOST'],
            port=os.environ['CONN_PORT']
        )        
        cursor = conn.cursor()
        if condicion:
            query = f"SELECT {columnas} FROM {tabla} WHERE {condicion};"
        else:
            query = f"SELECT {columnas} FROM {tabla};"        
        cursor.execute(query)
        if(multiple):
            resultados = cursor.fetchall()
        else:
            resultados=cursor.fetchone()
        cursor.close()
        conn.close()
        return resultados
    except (Exception, psycopg2.Error) as error:
        print("Error al conectar a la base de datos:", error)

def guardar_fondo(tabla, datos):
    try:        
        conn = psycopg2.connect(
            database=os.environ['CONN_DB'],
            user=os.environ['CONN_USER'],
            password=os.environ['CONN_PASWD'],
            host=os.environ['CONN_HOST'],
            port=os.environ['CONN_PORT']
        )         
        cursor = conn.cursor()        
        consulta = f"INSERT INTO {tabla} (est_emp_id,est_color_fondo_prin,est_color_prin_fondo,est_color_sec_fondo,est_color_ter_fondo) VALUES (%s, %s, %s, %s, %s)"        
        for dato in datos:
            valores = (dato['est_emp_id'], dato['est_color_fondo_prin'], dato['est_color_prin_fondo'],dato['est_color_sec_fondo'],dato['est_color_ter_fondo'])                       
            cursor.execute(consulta, valores)
        conn.commit()
        print("Datos insertados con éxito.")        
    except Exception as e:        
        conn.rollback()
        print(f"Error: {e}")
        return (f"Error: {e}")
    finally:        
        cursor.close()
        conn.close()
        return "Datos insertados con éxito."
    
def guardar_color(tabla, datos):
    try:        
        conn = psycopg2.connect(
            database=os.environ['CONN_DB'],
            user=os.environ['CONN_USER'],
            password=os.environ['CONN_PASWD'],
            host=os.environ['CONN_HOST'],
            port=os.environ['CONN_PORT']
        )         
        cursor = conn.cursor()        
        consulta = f"INSERT INTO {tabla} (est_emp_id,est_color_prin_color,est_color_sec_color,est_color_ter_color) VALUES (%s, %s, %s, %s)"        
        for dato in datos:
            valores = (dato['est_emp_id'], dato['est_color_prin_color'], dato['est_color_sec_color'],dato['est_color_ter_color'])                       
            cursor.execute(consulta, valores)
        conn.commit()
        print("Datos insertados con éxito.")        
    except Exception as e:        
        conn.rollback()
        print(f"Error: {e}")
        return (f"Error: {e}")
    finally:        
        cursor.close()
        conn.close()
        return "Datos insertados con éxito."
    
def actualizar_fondo(tabla, columna_condicion, valor_condicion, nuevos_valores):
    try:
        conn = psycopg2.connect(
            database=os.environ['CONN_DB'],
            user=os.environ['CONN_USER'],
            password=os.environ['CONN_PASWD'],
            host=os.environ['CONN_HOST'],
            port=os.environ['CONN_PORT']
        )        
        cursor = conn.cursor()        
        consulta = f"UPDATE {tabla} SET "
        for columna, nuevo_valor in nuevos_valores.items():
            consulta += f"{columna} = %s, "
        consulta = consulta[:-2]
        consulta += f" WHERE {columna_condicion} = %s"
        valores = list(nuevos_valores.values()) + [valor_condicion]
        cursor.execute(consulta, valores)        
        conn.commit()
        print("Fila actualizada con éxito.")
        return "Fila actualizada con éxito."
    except Exception as e:        
        conn.rollback()
        print(f"Error: {e}")
    finally:        
        cursor.close()
        conn.close()


def consultaEstilo(empresa):
    tabla="e_estilo"
    columnas="est_color_fondo_prin,est_color_prin_fondo,est_color_prin_color,est_color_sec_fondo,est_color_sec_color,est_color_ter_fondo,est_color_ter_color"
    condicion=f"est_emp_id={empresa}"
    resultados=consultar_tabla(tabla=tabla,columnas=columnas,condicion=condicion)
    return resultados

def consultaDominio(dominio):
    tabla="empresa"
    columnas="emp_id,emp_nombre"
    condicion=f"emp_dominio='{dominio}'"
    resultados=consultar_tabla(tabla=tabla,columnas=columnas,condicion=condicion)
    return resultados

def consultaSeguridadDominio(dominio,token,id):
    tabla="empresa"
    columnas="emp_id"
    condicion=f"emp_dominio='{dominio}' and emp_token='{token}' and emp_id='{id}'"
    resultados=consultar_tabla(tabla=tabla,columnas=columnas,condicion=condicion)
    return resultados

def guardarFondo(id,est_color_fondo_prin,est_color_prin_fondo,est_color_sec_fondo,est_color_ter_fondo):
    datos=[{'est_color_fondo_prin':est_color_fondo_prin,'est_color_prin_fondo':est_color_prin_fondo,'est_color_sec_fondo':est_color_sec_fondo,'est_color_ter_fondo':est_color_ter_fondo,'est_emp_id':id}]
    tabla="e_estilo"
    resultados=guardar_fondo(tabla=tabla,datos=datos)
    return resultados

def guardarColor(id,est_color_prin_color,est_color_sec_color,est_color_ter_color):
    datos=[{'est_color_prin_color':est_color_prin_color,'est_color_sec_color':est_color_sec_color,'est_color_ter_color':est_color_ter_color,'est_emp_id':id}]
    tabla="e_estilo"
    resultados=guardar_color(tabla=tabla,datos=datos)
    return resultados

def existenteEstilo(id):
    tabla="e_estilo"
    columnas='*'
    condicion=f"est_emp_id='{id}'"
    resultados=consultar_tabla(tabla=tabla,columnas=columnas,condicion=condicion)
    return resultados

def actualizarFondo(id,est_color_fondo_prin,est_color_prin_fondo,est_color_sec_fondo,est_color_ter_fondo):
    tabla="e_estilo"
    columna_condicion="est_emp_id"
    valor_condicion=f"{id}"
    nuevosvalores={'est_color_fondo_prin':est_color_fondo_prin,'est_color_prin_fondo':est_color_prin_fondo,'est_color_sec_fondo':est_color_sec_fondo,'est_color_ter_fondo':est_color_ter_fondo}
    resultados=actualizar_fondo(tabla=tabla,columna_condicion=columna_condicion,valor_condicion=valor_condicion,nuevos_valores=nuevosvalores)
    return resultados

def actualizarColor(id,est_color_prin_color,est_color_sec_color,est_color_ter_color):
    tabla="e_estilo"
    columna_condicion="est_emp_id"
    valor_condicion=f"{id}"
    nuevosvalores={'est_color_prin_color':est_color_prin_color,'est_color_sec_color':est_color_sec_color,'est_color_ter_color':est_color_ter_color}
    resultados=actualizar_fondo(tabla=tabla,columna_condicion=columna_condicion,valor_condicion=valor_condicion,nuevos_valores=nuevosvalores)
    return resultados

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = os.environ["JWT_SECRET_KEY"]
jwt = JWTManager(app)
CORS(app)

@app.route('/bot-question', methods=['POST'])
@jwt_required()
def response_question():
    question = request.get_json()
    return jsonify(queryModel(question['question']))

@app.route('/get-audio', methods=['POST'])
@jwt_required()
def get_audio():
    return jsonify(saveFileAudio(request))

@app.route("/token", methods=["POST"])
def create_token():
    urlDominio = request.json.get("domain", None)
    dbDominio=consultaDominio(urlDominio)    
    if (dbDominio!=None):
        access_token = create_access_token(identity=dbDominio[1])
        return jsonify({ "token": access_token, "identity": dbDominio[1], "id":dbDominio[0]})
    else:
        abort(403)

@app.route('/consult-style', methods=['POST'])
def consult_style():
    id = request.json.get("id", None)
    respuesta=consultaEstilo(id)
    if(respuesta!=None):
        return jsonify(respuesta)
    else:
        abort(403)

@app.route('/save-background', methods=['POST'])
def save_background():
    est_emp_id = request.json.get("est_emp_id", None)
    est_color_fondo_prin = request.json.get("est_color_fondo_prin", None)
    est_color_prin_fondo = request.json.get("est_color_prin_fondo", None)    
    est_color_sec_fondo = request.json.get("est_color_sec_fondo", None)    
    est_color_ter_fondo = request.json.get("est_color_ter_fondo", None)
    verificar=existenteEstilo(est_emp_id)    
    if(verificar!=None):
        resultado=actualizarFondo(id=est_emp_id,est_color_fondo_prin=est_color_fondo_prin,est_color_prin_fondo=est_color_prin_fondo,est_color_sec_fondo=est_color_sec_fondo,est_color_ter_fondo=est_color_ter_fondo)
    else:
        resultado=guardarFondo(id=est_emp_id,est_color_fondo_prin=est_color_fondo_prin,est_color_prin_fondo=est_color_prin_fondo,est_color_sec_fondo=est_color_sec_fondo,est_color_ter_fondo=est_color_ter_fondo)
    if(resultado!=None):
        return jsonify(resultado)
    else:
        abort(403)

@app.route('/save-color', methods=['POST'])
def save_color():
    est_emp_id = request.json.get("est_emp_id", None)    
    est_color_prin_color = request.json.get("est_color_prin_color", None)
    est_color_sec_color = request.json.get("est_color_prin_color", None)
    est_color_ter_color = request.json.get("est_color_prin_color", None)
    verificar=existenteEstilo(est_emp_id)
    if(verificar!=None):
        resultado=actualizarColor(id=est_emp_id,est_color_prin_color=est_color_prin_color,est_color_sec_color=est_color_sec_color,est_color_ter_color=est_color_ter_color)
    else:
        resultado=guardarColor(id=est_emp_id,est_color_prin_color=est_color_prin_color,est_color_sec_color=est_color_sec_color,est_color_ter_color=est_color_ter_color)
    if(resultado!=None):
        return jsonify(resultado)
    else:
        abort(403)

@app.route('/verify-auth',methods=['POST'])
def verify_auth():
    urlDominio = request.json.get("domain", None)
    token = request.json.get("token", None)
    id = request.json.get("id", None)
    resultado=consultaSeguridadDominio(dominio=urlDominio,token=token,id=id)
    if (resultado!=None):
        return jsonify(resultado)
    else:
        abort(403)



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)