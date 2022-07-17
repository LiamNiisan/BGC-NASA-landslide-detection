# syntax=docker/dockerfile:1
FROM python:3.9
WORKDIR .
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN python -m nltk.downloader stopwords
RUN python -m spacy download en_core_web_sm
RUN ["python", "-c", "from flair.models import SequenceTagger\nSequenceTagger.load(\"ner-ontonotes-fast\")"]
COPY . .
CMD ["python", "src/main.py"]