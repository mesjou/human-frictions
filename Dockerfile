FROM python:3.7.10

ADD human_friction/train.py /

COPY requirements.txt .
RUN pip3 install --upgrade pip \
 && pip3 install --default-timeout=5000 -r requirements.txt

COPY . .

CMD [ "python", "./train.py" ]
