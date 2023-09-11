From python:3.8

# environment variable 
ENV DockerHOME=/home/app/report

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
RUN pip install -r requirements.txt
ARG port=5000
EXPOSE $port:$port
CMD ["flask", "--app", "main.py", "run","--host=0.0.0.0","--port=5000"]