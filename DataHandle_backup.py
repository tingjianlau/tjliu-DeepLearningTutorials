#!/usr/bin/env python
# -*- coding: UTF-8 -*-

##########################################################################
#	> File Name: data-handle.py
#	> Author: Tingjian Lau
#	> Mail: tjliu@mail.ustc.edu.cn
#	> Created Time: 2016/09/29
#	> Detail: 
#########################################################################

import numpy as np
import theano
import theano.tensor as T

class DataHandle(object):
    def __init__(self, row_start=1, row_end=5):
        self.row_start = row_start 
        self.row_end = row_end 

    def load_data(self, data_path, seq_length=10, axis1=0, axis2=0, del_table_header = 1, delimiter='\t'):
        '''
        params:
        ------
        seq_length: 将seq_length条数据拼接成一条
        axis1,axis2: 通过比较第seq_length个样本的第aixs2维的值与
                    第seq_length+1个样本的第axis1的值得到类号
        del_table_header: 是否删除第一行，即表头
        delimiter: 数据的分隔符
        return: 拼接好之后的数据集和类号
        '''

        print("Loading data from ", data_path)
        data = open(data_path, 'r')
        data_set_x = []
        for line in data:
            data_set_x.append(line.split('\n')[0].split(delimiter)[self.row_start:self.row_end])
        data.close()
        if del_table_header:
            data_set_x = data_set_x[1::]

        data_set_y = []
        new_len = len(data_set_x)-seq_length
        for i in range(new_len):
            if data_set_x[i+seq_length-1][axis1] > data_set_x[i+seq_length][axis2]:
                data_set_y.append(0)
            else:
                data_set_y.append(1)

            for j in range(seq_length-1):
                data_set_x[i].extend(data_set_x[i+j+1])

        return data_set_x[0:new_len], data_set_y

    def minmax_norm(self, data_set):
        data = np.zeros((len(data_set), len(data_set[0])))  

        for i in range(len(data_set)):
            for j in range(len(data_set[0])):
                data[i][j] = data_set[i][j]

        max = np.amax(data, axis=0)
        min = np.amin(data, axis=0)

        for i in range(data.shape[0]):
            data[i] = (data[i] - min) / (max - min)

        return data

    def shared_dataset(self, data_set_x, data_set_y, borrow=True):
        shared_x = theano.shared(np.asarray(data_set_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_set_y,dtype=theano.config.floatX),borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

        
