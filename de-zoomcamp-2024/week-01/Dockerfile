FROM python:3.9

RUN pip install pandas
RUN pip install numpy

WORKDIR /the/workdir/path
COPY pipeline.py pipeline.py

ENTRYPOINT ["python", "pipeline.py"]