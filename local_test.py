# coding:utf-8
# ---------本脚本用于测试svm_classifier---------

from __future__ import unicode_literals

import os
import codecs
from glob import glob
from svm_classifier import SVMTextClassifier

types = ['D001002001', 'D001002002', 'D001002003', 'D001003001', 'D001003002', 'D001003003', 'D001004001',
         'D001004002', 'D001004003', 'NOTICE', 'OTHER', 'T004001001', 'T004001002', 'T004004001',
         'T004004002', 'T004004003', 'T004004004', 'T004004005', 'T004005001', 'T004005002', 'T004006001',
         'T004006005', 'T004009001', 'T004009002', 'T004019001', 'T004019003', 'T004021008', 'T004022018',
         'T004023007']

# filedir = '/home/zhwpeng/abc/text_classify/data/0412/raw/test_data3/T004019001/2159233.txt'
# with codecs.open(filedir, 'r', encoding='utf-8') as fr:
#     data = fr.read()
#
#
# typ = classifier.classify(data)
# print typ

classifier = SVMTextClassifier()

data_dir = "/home/zhwpeng/abc/text_classify/data/0412/raw/train_data/"
# type_c = types[13]
type_c = types[6]
txt_files = glob(os.path.join(data_dir, type_c, '*.txt'))
wrong_nums = 0
for txt_file in txt_files:
    with codecs.open(txt_file, 'r', encoding='utf-8') as fr:
        txt_content = fr.read()
    txt_type = classifier.classify(txt_content)
    if txt_type != type_c:
        wrong_nums += 1
        print "file {} is {} type, but has been classified to {}".format(txt_file.split('/')[-1], type_c, txt_type)
print "folder {} has {} files, wrong classified {} files".format(type_c, len(txt_files), wrong_nums)


