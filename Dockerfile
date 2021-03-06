# syntax=docker/dockerfile:1

FROM tensorflow/tensorflow:2.5.0

WORKDIR .

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]