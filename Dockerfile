# syntax=docker/dockerfile:1

FROM python:3.7-slim-buster

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN /bin/bash -c 'python3 -m pip install -r requirements.txt'
RUN /bin/bash -c 'python3 -m spacy download en_core_web_md'
RUN /bin/bash -c 'python3 -m spacy link en_core_web_md en --force;'
RUN /bin/bash -c 'python3 -m pip install rasa_nlu==0.14'
RUN /bin/bash -c 'python3 -m pip install scikit-learn==0.22.2.post1'

COPY . .

CMD [ "flask", "run", "--host=0.0.0.0"]
EXPOSE 5000
