From python:3.8

# environment variable 
ENV DockerHOME=/home/app/api

# directory to work
RUN mkdir -p $DockerHOME  

# where your code lives  
WORKDIR $DockerHOME 

# environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1  
ENV PYTHONIOENCODING UTF-8

# update dependencias of python 
RUN pip install --upgrade pip

# install libraries

RUN apt-get update && apt-get install -y -q --no-install-recommends
RUN apt-get install -y build-essential
RUN apt-get install -y libpq-dev
run apt --fix-broken install
COPY . $DockerHOME
RUN pip install numpy==1.24.4
RUN pip install openai[embeddings]==0.27.6
RUN pip install fsspec==2023.4.0
RUN pip install FuzzyTM==2.0.5
RUN pip install blosc2~=2.0.0
RUN pip install cython>=3.0.2
RUN pip install langchain==0.0.279
RUN pip install pypdf==3.8.1
RUN pip install tiktoken==0.3.3
RUN pip install faiss-cpu==1.7.4
RUn pip install unstructured==0.6.2
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install chromadb==0.3.21
RUN pip install llama-index==0.6.1
RUN pip install flask
RUN pip install flask_cors
RUN pip install jsonschema
RUN pip install flask-jwt-extended
ARG port=5000
EXPOSE $port:$port
CMD ["flask", "--app", "main.py", "run","--host=0.0.0.0","--port=5000"]