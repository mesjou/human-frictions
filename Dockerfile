FROM python:3.7.10

ADD src/train.py /

COPY requirements.txt .
RUN pip3 install --upgrade pip \
 && pip3 install -r requirements.txt

COPY . /app

CMD [ "python", "./train.py" ]
