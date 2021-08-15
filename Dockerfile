FROM ubuntu:20.04

RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -y && \
    apt-get install -y python3.6 python3-pip && \
    apt-get update -y && \
    apt-get install -y python3-setuptools python3-wheel sqlite3 && \
    python3 --version && \
    pip3 --version && \
    sqlite3 --version

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip3 install -r requirements.txt

VOLUME /mapi

COPY . /

ENTRYPOINT [ "python3" ]

CMD [ "app/app.py" ]
