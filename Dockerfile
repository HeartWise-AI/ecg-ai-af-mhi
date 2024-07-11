FROM tensorflow/tensorflow:2.13.0-gpu

COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["/bin/sh", "runall.sh"]
