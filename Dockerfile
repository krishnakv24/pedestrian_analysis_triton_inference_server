FROM nvcr.io/nvidia/tritonserver:24.08-py3

RUN pip install --no-cache-dir tritonclient[grpc] numpy

EXPOSE 8001 8002

CMD ["tritonserver", \
     "--model-repository=/models", \
     "--grpc-port=8001", \
     "--allow-http=false", \
     "--allow-grpc=true", \
     "--allow-metrics=true", \
     "--metrics-port=8002", \
     "--log-verbose=1"]
