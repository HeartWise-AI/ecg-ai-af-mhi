FROM python:3.8

COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

VOLUME ["/xml-data"]

CMD ["/bin/sh", "runall.sh"]