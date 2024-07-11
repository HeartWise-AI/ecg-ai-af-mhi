FROM tensorflow/tensorflow:2.13.0-gpu

COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements_docker.txt

CMD ["/bin/sh", "runall.sh"]
