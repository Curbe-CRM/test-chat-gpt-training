import pymongo
from bson import ObjectId
import openai
import os
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index import (
    SimpleDirectoryReader,
    PromptHelper,
    GPTVectorStoreIndex,
    LLMPredictor,
    StorageContext,
    load_index_from_storage,
    ServiceContext, SQLDatabase
)
import gradio as gr
import sys
from flask import Flask, request
import json
import bson.json_util as json_util


from llama_index.indices.struct_store import NLSQLTableQueryEngine
from sqlalchemy import create_engine, inspect

from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.readers.file.base import SimpleDirectoryReader



os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = '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'

# openai.api_key = ''
# chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
#                                               messages=[{"role": "user", "content": "Cual es la mejor moto del mundo"}])

# print('chat_completion')
#
# engine = create_engine(connection_uri)
# llm = OpenAI(
#     openai_api_key='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',
#     temperature=0.5,
#     model_name="text-davinci-003",
#     max_tokens=512,
# )
# service_context = ServiceContext.from_defaults(llm=llm)
# sql_database = SQLDatabase(engine)
# inspector = inspect(engine)
# table_names = inspector.get_table_names()
# print(table_names)

# def chat_to_pytohn sql(question: str):
#     query_engine = NLSQLTableQueryEngine(
#         sql_database=sql_database,
#         tables=table_names,
#         synthesize_response=True,
#         service_context=service_context
#     )
#     try:
#         response = query_engine.query(question)
#         response_md = str(response)
#         sql = response.metadata['sql_query']
#     except Exception as ex:
#         response_md = 'Error'
#         sql = f'ERROR: {str(ex)}'
#     return response_md


def construct_index(directory_path):
    """
    Construct index data from directory path.
    :param directory_path: The path of the directory.
    :return: A VectorStoreIndex.
    """
    max_input_size = 600
    num_outputs = 60
    max_chunk_overlap = 0
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit
    )
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            openai_api_key='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',
            temperature=1,
            model_name="gpt-3.5",
            max_tokens=num_outputs,
        )
    )

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex(
        documents,
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper
    )
    index.storage_context.persist(persist_dir='docs/train')
    return index

def chatbot(input_text):
    storage_context = StorageContext.from_defaults(persist_dir='docs/train')
    index = load_index_from_storage(storage_context)
    response = index.as_query_engine()
    return response.query(input_text).response

def mongoRead():
    client = pymongo.MongoClient("mongodb://corp_usr:corp2021@10.124.0.2")
    db = client['mailcomdev']
    collection = db['e_report_sale']
    aggregation_pipeline = [
    {
        '$match': {
            'ref_periodo': 2022, 
            'ref_empresa': 1, 
            'ref_mes': 1
        }
    }, {
        '$project': {
            'nombre_almacen': '$ref_almacen_nombre', 
            'ciudad_almacen': '$ref_almacen_ciudad', 
            'canal_distribucion': '$ref_can_nombre', 
            '_id': 0, 
            'nombre_cliente': '$ref_cli_nombre', 
            'cantidad_compra': '$ref_doc_cantidad', 
            'nombre_producto': '$ref_pro_nombre', 
            'total_venta': '$ref_total', 
            'atributo_1_producto': '$ref_pro_id_nivel1', 
            'ref_pro_id_nivel2': 1, 
            'ref_pro_id_nivel3': 1, 
            'ref_pro_atributo4': 1, 
            'ref_pro_atributo5': 1
        }
    }
]
    documentos = collection.aggregate(aggregation_pipeline)
    for documento in documentos:
        with open('data.json', 'w') as f:
            f.write(json_util.dumps(documento))
    client.close()
    
#mongoRead()

#construct_index("data/indice")

app = Flask(__name__)


@app.route('/bot-question', methods=['POST'])
def response_question():
   question = request.get_json()
   return chatbot(question['question'])
