#!/usr/bin/env python
# -*- coding: UTF-8 -*-

##########################################################################
#	> File Name: StockDataHandleClass.py
#	> Author: Tingjian Lau
#	> Mail: tjliu@mail.ustc.edu.cn
#	> Created Time: 2016/09/23
#	> Detail: 
#########################################################################

class StockDataHandle
    def my_load_data(dataset):
        print('... loading data') data = open(dataset, 'r')
        prices = []
        for line in data:
            prices.append(line.split('\n')[0].split('\t')[1:5])
        return prices #del title

    def data_prehandle(data_set_x, SEQ_LENGTH, axis1 = 0, axis2 = 0):
        data_set_y = []
        new_len = len(data_set_x)-SEQ_LENGTH
        for i in range(new_len):
            if data_set_x[i+SEQ_LENGTH-1][axis1] > data_set_x[i+SEQ_LENGTH][axis2]:
                data_set_y.append(0)
            else:
                data_set_y.append(1)

            for j in range(SEQ_LENGTH-1):
                data_set_x[i].extend(data_set_x[i+j+1])

        return data_set_x[0:new_len], data_set_y

    def minmax_norm(data_set):
        data = np.zeros((len(data_set), len(data_set[0])))  

        for i in range(len(data_set)):
            for j in range(len(data_set[0])):
                data[i][j] = data_set[i][j]

        max = np.amax(data, axis=0)
        min = np.amin(data, axis=0)

        for i in range(data.shape[0]):
            data[i] = (data[i] - min) / (max - min)

        return data

    def shared_dataset(data_set_x, data_set_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_set_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_set_y,dtype=theano.config.floatX),borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

