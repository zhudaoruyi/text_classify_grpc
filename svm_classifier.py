# coding=utf-8

from __future__ import absolute_import

import logging
import re
import os
# import jieba
import pickle
import numpy as np
import pandas as pd
# from pyhanlp import *
import jpype
from jpype import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

Label_Line = ['D001002001', 'D001002002', 'D001002003', 'D001003001', 'D001003002', 'D001003003', 'D001004001',
              'D001004002', 'D001004003', 'NOTICE', 'OTHER', 'T004001001', 'T004001002', 'T004004001',
              'T004004002', 'T004004003', 'T004004004', 'T004004005', 'T004005001', 'T004005002', 'T004006001',
              'T004006005', 'T004009001', 'T004009002', 'T004019001', 'T004019003', 'T004021008', 'T004022018',
              'T004023007']

log = logging.getLogger(__name__)


class BaseClassifier(object):
    def classify(self, text):
        raise NotImplemented

    def classify_batch(self, text, batch_size):
        raise NotImplemented


class SVMTextClassifier(BaseClassifier):
    def __init__(self,
                 vocabulary='abcChinese_jieba.txt',
                 model='train_tmp_ht03/new_model/svm_c23ch1000s0.7999.pkl',
                 all_features='train_tmp_ht03/all_features_dir/all_features.pkl',
                 selected_features='train_tmp_ht03/features_index_by_ch2_1000/features_index_by_ch2.npy'):
        self.model = pickle.load(open(model, 'rb'))
        self.all_features = pickle.load(open(all_features, 'rb'))
        self.selected_features = np.load(selected_features)
        # self.vocabulary_path = vocabulary
        # jieba.load_userdict(self.vocabulary_path)

        hanlplibpath = "/home/zhwpeng/venv2/lib/python2.7/site-packages/pyhanlp/static/"
        # # hanlplibpath = "/home/zhwpeng/hanlp/"
        jpype.startJVM(getDefaultJVMPath(),
                 "-Djava.class.path=" + hanlplibpath + "hanlp-portable-1.6.0.jar:" + hanlplibpath + "data",
                 # "-Djava.class.path="+hanlplibpath+"hanlp-1.2.8.jar:"+hanlplibpath+"data",
                 "-Xms1g", "-Xmx1g")
        self.HanLP = jpype.JClass('com.hankcs.hanlp.HanLP')
        # if jpype.isJVMStarted():
        # if not jpype.isThreadAttachedToJVM():
        # self.HanLP = HanLP

    def is_chinese(self, uchar):
        """判断一个unicode是否是汉字"""
        X, Y = [u'\u4e00', u'\u9fa5']  # unicode 前面加u
        if uchar >= X and uchar <= Y:
            return True
        else:
            return False

    def is_number(self, uchar):
        """判断一个unicode是否是数字"""
        X, Y = [u'\u0030', u'\u0039']
        if uchar >= X and uchar <= Y:
            return True
        else:
            return False

    def is_alphabet(self, uchar):
        """判断一个unicode是否是英文字母"""
        a, z, A, Z = [u'\u0041', u'\u005a', u'\u0061', u'\u007a', ]
        if (uchar >= a and uchar <= z) or (uchar >= A and uchar <= Z):
            return True
        else:
            return False

    def is_other(self, uchar):
        """判断是否非汉字，数字和英文字符"""
        if not (self.is_chinese(uchar) or self.is_number(uchar) or self.is_alphabet(uchar)):
            return True
        else:
            return False

    def separate_words_by_hanlp(self, input_text, stopwords):
        out_text = []
        # seg_list = hanlp_cut(input_text)  # 通过grpc调用hanlp
        jpype.attachThreadToJVM()
        seg_list = self.HanLP.segment(input_text)  # 直接调用pyhanlp
        # seg_list = jieba.cut(input_text)
        for term in seg_list:
            word = term.word
            # word = term
            if self.is_other(word):
                continue
            if self.is_number(word):
                continue
            if self.is_alphabet(word):
                continue
            try:
                word = word.encode('UTF-8')
            except:
                continue
            if word not in stopwords:
                if word != ' ':
                    out_text.append(word)
                    out_text.append(' ')
        return out_text

    def preprocess(self, fulltext):
        out_text = []
        stopwords_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'stopwords.txt')
        stopwords = []
        # 获取停用词
        with open(stopwords_path) as fi:
            for line in fi.readlines():
                stopwords.append(line.strip())
        fulltext = fulltext.replace('\n', '').replace(' ', '')
        fulltext = re.sub("[\s+：（）“”，■？、！…,/'《》<>!?_——=()-]+".decode("UTF-8", "ignore"),
                          "".decode("UTF-8", "ignore"), fulltext)
        fulltext = fulltext.split(u'。')
        for line in fulltext:
            line = self.separate_words_by_hanlp(line, stopwords)
            if not len(line):
                continue
            for word in line:
                if word != ' ':
                    out_text.append(word)
        fulltext = ' '.join(out_text)
        return fulltext

    def get_tfidf(self, count_v0, train_texts):
        counts_train = count_v0.fit_transform(train_texts)

        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(counts_train)

        feature_names = count_v0.get_feature_names()  # 关键字
        count_v0_df = pd.DataFrame(counts_train.toarray())
        tfidf_df = pd.DataFrame(tfidf.toarray())
        return count_v0_df, tfidf_df, feature_names

    def get_x_test(self, count_v0_df, feature_names, selected_features):
        count_v0_tfsx_df = count_v0_df.ix[:, selected_features]  # tfidf筛选后的词向量矩阵
        df_columns = pd.Series(feature_names)[selected_features]

        def guiyi(x):
            x[x > 1] = 1
            return x

        tfidf_df_1 = count_v0_tfsx_df.apply(guiyi)
        return tfidf_df_1, df_columns

    def post_procession(self, predictions):
        pass

    def classify(self, fulltext):
        fulltext = self.preprocess(fulltext)
        count_text = CountVectorizer(decode_error='replace', vocabulary=self.all_features)  # 特征词库
        count_v0_df, _, feature_names = self.get_tfidf(count_text, [fulltext])  # count_v0_df这是测试文本的词频，后面两个跟训练集保持一致
        tfidf_df_1, _ = self.get_x_test(count_v0_df, feature_names, self.selected_features)  # 筛选后的特征词
        predictions = self.model.predict(tfidf_df_1)
        return Label_Line[predictions[0]]

    def classify_batch(self, fulltext, batch_size=100):
        pass
