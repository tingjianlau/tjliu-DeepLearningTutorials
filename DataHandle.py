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
    def __init__(self, seq_length=10, row_interval=[1, 5], axes=[0, 0],
                delimiter='\t', del_table_header=True):
        self._row_start = row_interval[0]
        self._row_end = row_interval[1]
        self._seq_length = seq_length
        self._axis1=axes[0]
        self._axis2=axes[1]
        self._delimiter=delimiter
        self._del_table_header=del_table_header

    '''
    def load_data(self, data_path):
        params:
        ------
        seq_length: 将seq_length条数据拼接成一条
        axis1,axis2: 通过比较第seq_length个样本的第aixs2维的值与
                    第seq_length+1个样本的第axis1的值得到类号
        del_table_header: 是否删除第一行，即表头
        delimiter: 数据的分隔符
        return: 拼接好之后的数据集和类号

        print("Loading data from {} ...".format(data_path))
        data = open(data_path, 'r')
        data_set_x = []
        for line in data:
            data_set_x.append(line.split('\n')[0].split(self._delimiter)[self._row_start:self._row_end])
        data.close()
        if self._del_table_header:
            data_set_x = data_set_x[1::]

        data_set_y = []
        new_len = len(data_set_x)-self._seq_length
        for i in range(new_len):
            if data_set_x[i+self._seq_length-1][self._axis1] > data_set_x[i+self._seq_length][self._axis2]:
                data_set_y.append(0)
            else:
                data_set_y.append(1)

            for j in range(self._seq_length-1):
                data_set_x[i].extend(data_set_x[i+j+1])

        return data_set_x[0:new_len], data_set_y
    '''
    def joint_x_gen_y(self, data_set_x):
        # 用于日线数据的拼接和生成类号
        data_set_y = []
        new_len = len(data_set_x)-self._seq_length
        for i in range(new_len):
            if data_set_x[i+self._seq_length-1][self._axis1] > data_set_x[i+self._seq_length][self._axis2]:
                data_set_y.append(0)
            else:
                data_set_y.append(1)

            for j in range(self._seq_length-1):
                data_set_x[i].extend(data_set_x[i+j+1])

        return data_set_x[0:new_len], data_set_y

    def load_data(self, data_path):
        print("Loading data from {} ...".format(data_path))
        data = open(data_path, 'r')
        data_set_x = []
        for line in data:
            data_set_x.append(line.split('\n')[0].split(self._delimiter)[self._row_start:self._row_end])
        data.close()
        if self._del_table_header:
            data_set_x = data_set_x[1::]

        return data_set_x

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

    def shared_dataset_x(self, data_set_x, borrow=True):
        shared_x = theano.shared(np.asarray(data_set_x, dtype=theano.config.floatX), borrow=borrow)

        return shared_x

    def shared_dataset_y(self, data_set_y, borrow=True):
        shared_y = theano.shared(np.asarray(data_set_y,dtype=theano.config.floatX),borrow=borrow)

        return T.cast(shared_y, 'int32')

    def get_datasets(self, datapath):
        '''
        params:
        ------
        datapath:   list
            加载的路径列表
        return: 
        ------
        以列表的形式返回所有的数据集和类号
        '''
        unshared_datasets = []
        shared_datasets = []
        for i in range(len(datapath)):
            #data_set_x, data_set_y = self.load_data(datapath[i])
            data_set_x = self.load_data(datapath[i])
            data_set_x, data_set_y = self.joint_x_gen_y(data_set_x)
            unshared_datasets.append((data_set_x, data_set_y))
            # 归一化
            data_set_x = self.minmax_norm(data_set_x)
            # 转化为共享变量
            data_set_x = self.shared_dataset_x(data_set_x)
            data_set_y = self.shared_dataset_y(data_set_y)
            shared_datasets.append((data_set_x, data_set_y))

        return shared_datasets, unshared_datasets


class DataHandle_minute(DataHandle):
    def __init__(self, seq_length=10, row_interval=[1, 5], axes=[0, 0],
                delimiter='\t', del_table_header=True, row_mid=2):
        # 调用父类的init函数
        DataHandle.__init__(self, seq_length, row_interval, axes, delimiter,
                del_table_header)
        self._row_mid = row_mid
    

    def joint_x_gen_daily(self, data_set_x):
        '''
        在处理分钟数据时，data_set_x的前两唯是日期和分钟时刻
        return: 拼接好的分钟数据和计算后的日线数据
        '''
        data_range = []
        data_range.append(0)
        for i in xrange(len(data_set_x)):
            if data_set_x[i][0] != data_set_x[data_range[-1]][0]:
                data_range.append(i)
        data_range.append(len(data_set_x))


        data_daily = [] # 存放由分钟时刻数据计算出的日线数据
        data_minute = []    # 存放拼接后的每天的分钟时刻数据
        for i in range(len(data_range)-1):
            open_price = data_set_x[data_range[i]][self._row_mid]
            close_price = data_set_x[data_range[i+1]-1][self._row_mid+3]
            high_price = 0
            low_price = 100000
            minute_item = []    # 存放拼接后某一天的分钟时刻数据
            for j in range(data_range[i], data_range[i+1]):
                for k in range(self._row_mid, self._row_end):
                    minute_item.append(float(data_set_x[j][k]))
                if high_price < float(data_set_x[j][self._row_mid+1]):
                    high_price = float(data_set_x[j][self._row_mid+1])
                if low_price > float(data_set_x[j][self._row_mid+2]):
                    low_price = float(data_set_x[j][self._row_mid+2])
            data_minute.append(minute_item)
            # 当天的数据
            data_daily.append([float(open_price), high_price, float(close_price), low_price]) 

        for item in data_minute:
            assert(len(item)%48==0)
            break
        with open('tmp.txt', 'wb') as f:
            for item in data_daily:
                f.write('%s\n' % item)

        with open('tmp2.txt', 'wb') as f:
            for item in data_minute:
                f.write('%s\n' % item)
        '''
        暂时不考虑数据损坏的情况，如出现11:35,15:05
        bad_data = []
        for i in range(len(data_range)-1):
            if data_range[i+1] - data_range[i] != 48:
                print(data_range[i+1])
        #print(len(data_range))
        '''


        return data_daily, data_minute 

    def get_datasets(self, datapath):
        '''
        params:
        ------
        datapath:   list
            加载的路径列表
        return: 
        ------
        unshared_datasets: 为LSTM准备的数据集
        shared_datasets: 为RBMs准备的数据集
        以列表的形式返回所有的数据集和类号
        '''

        unshared_datasets = []
        shared_datasets = []
        for i in range(len(datapath)):
            data_set_x = self.load_data(datapath[i])
            data_daily, data_minute = self.joint_x_gen_daily(data_set_x)
            data_daily_x, data_daily_y = self.joint_x_gen_y(data_daily)
            unshared_datasets.append((data_minute, data_daily_y)) 
            # 归一化
            data_minute = self.minmax_norm(data_minute)
            # 转化为共享变量
            data_minute = self.shared_dataset_x(data_minute)
            shared_datasets.append(data_minute)

        return shared_datasets, unshared_datasets
        


   
