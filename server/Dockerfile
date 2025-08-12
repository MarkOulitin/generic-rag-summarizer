FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /home
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ./src ./

CMD ["python", "server.py"]