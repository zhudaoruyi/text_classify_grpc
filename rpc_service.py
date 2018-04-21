# coding:utf-8
from __future__ import unicode_literals

import grpc
import time
import tc_pb2
import tc_pb2_grpc
from concurrent import futures
from svm_classifier import SVMTextClassifier


class TextClassifyServicer(tc_pb2_grpc.GetClassifierServicer):
    def __init__(self):
        self.classifier = SVMTextClassifier()

    def TextClassify(self, request, context):
        response = tc_pb2.Data()
        response.value = self.classifier.classify(request.value)
        return response


# create a gRPC server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

# use the generated function `add_CalculatorServicer_to_server`
# to add the defined class to the created server
tc_pb2_grpc.add_GetClassifierServicer_to_server(
    TextClassifyServicer(), server
)

# listen on port 50051
print('Starting server. Listening on port 50051.')
server.add_insecure_port('[::]:50051')
server.start()

# since server.start() will not block,
# a sleep-loop is added to keep alive
try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)
