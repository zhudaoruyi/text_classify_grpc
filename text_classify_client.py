# coding:utf-8
from __future__ import unicode_literals

import grpc
import codecs
import tc_pb2
import tc_pb2_grpc
# open a gRPC channel
channel = grpc.insecure_channel('localhost:50051')

# create a stub (client)
stub = tc_pb2_grpc.GetClassifierStub(channel)

# create a valid request message
filedir = '/home/zhwpeng/abc/text_classify/data/0412/raw/test_data3/T004019001/2159233.txt'
with codecs.open(filedir, 'r', encoding='utf-8') as fr:
    data = fr.read()
response = stub.TextClassify(tc_pb2.Data(value=data))

print(response.value)
