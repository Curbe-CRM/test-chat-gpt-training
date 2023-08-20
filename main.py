import openai
import os
from langchain import OpenAI
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

from llama_index.indices.struct_store import NLSQLTableQueryEngine
from sqlalchemy import create_engine, inspect

db_user = ''
db_password = ""
db_host = ""
db_name = ""
db_port = ""
# connection_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
connection_uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# This is a sample Python script.

# os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''

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

# def chat_to_sql(question: str):
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
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 1
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit
    )
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            openai_api_key='',
            temperature=0.5,
            model_name="text-davinci-003",
            max_tokens=num_outputs,
        )
    )

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex(
        documents,
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper
    )
    index.storage_context.persist(persist_dir='docs/index.json')
    return index


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def chatbot(input_text):
    storage_context = StorageContext.from_defaults(persist_dir='docs/index.json')
    index = load_index_from_storage(storage_context)
    response = index.as_query_engine()
    return response.query(input_text).response


# iface = gr.Interface(fn=chatbot,
 #                    inputs=gr.inputs.Textbox(lines=3, label="Enter your text"),
  #                  outputs="text",
 #                    title="My AI Chatbot")

# construct_index("docs")
#iface.launch(share=True)

# print(construct_index('docs'))

app = Flask(__name__)


@app.route('/bot-question', methods=['POST'])
def response_question():
    question = request.get_json()
    return chatbot(question['question'])
