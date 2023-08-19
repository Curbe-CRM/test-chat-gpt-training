import openai
import os
from langchain import OpenAI
from llama_index import (
    SimpleDirectoryReader,
    PromptHelper,
    GPTVectorStoreIndex,
    LLMPredictor,
    StorageContext,
    load_index_from_storage
)
import gradio as gr
import sys

# This is a sample Python script.

# os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''

# openai.api_key = ''
# chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
#                                               messages=[{"role": "user", "content": "Cual es la mejor moto del mundo"}])

print('chat_completion')


def construct_index(directory_path):
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
            odel_name="text-davinci-003",
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


iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=3, label="Enter your text"),
                     outputs="text",
                     title="My AI Chatbot")

index = construct_index("docs")
iface.launch(share=True)

# print(construct_index('docs'))
