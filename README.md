# Test-Chatbot-Gpt-Training
Trianing GTP Chatbot test. Data from PDFs,JSON.
## PIP Requirements
- numpy==1.24.4
- openai[embeddings]==0.27.6
- fsspec==2023.4.0
- FuzzyTM==2.0.5
- blosc2~=2.0.0
- cython>=3.0.2
- langchain==0.0.155
- pypdf==3.8.1
- tiktoken==0.3.3
- faiss-cpu==1.7.4
- unstructured==0.6.2
- chromadb==0.3.21
- llama-index==0.6.1
- flask
- flask_core
- jsonschema
## OS Requirements
- libpq-dev
- build-essential
- git
- conda
- python==3.8
## Commands
flask --app main.py run --host=0.0.0.0 --port=5000
