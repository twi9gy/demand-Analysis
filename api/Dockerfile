FROM python:3.6

#update
RUN apt-get update

#install requirements
COPY ./requirements.txt /tmp/requirements.txt
WORKDIR /tmp
RUN pip3 install -r requirements.txt

#copy api
COPY . /api
WORKDIR /

CMD ["gunicorn", "-b", ":5000", "--reload", "api.wsgi:app", "--timeout=300"]