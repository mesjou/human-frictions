FROM python:3.7.10

ADD human_friction/train.py /

COPY requirements.txt .
RUN pip3 install --upgrade pip \
 && pip3 install -r requirements.txt

COPY . /human_friction/

CMD [ "python", "./train.py" ]
