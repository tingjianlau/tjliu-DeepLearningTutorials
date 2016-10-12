#!/usr/bin/env python 
# -*- coding: UTF-8 -*-

##########################################################################
#	> File Name: my_DBN.py
#	> Author: Tingjian Lau
#	> Mail: tjliu@mail.ustc.edu.cn
#	> Created Time: 2016/09/20
#	> Detail: 
#########################################################################

from __future__ import print_function, division
import os
import sys
import timeit

import numpy as np
import numpy

import pickle
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from logistic_sgd import LogisticRegression, load_data
from DBN import DBN
from DataHandle import DataHandle

datapath = ['data_minute/000002/data_000002_train', 'data_minute/000002/data_000002_val', 'data_minute/000002/data_000002_test'] 
data_advanced_features_path = ['data_minute/000002/data_features/000002_train_finetune_epoch_3000_size_20_axis_0_0', 'data_minute/000002/data_features/000002_val_finetune_epoch_3000_size_20_axis_0_0', 'data_minute/000002/data_features/000002_test_finetune_epoch_3000_size_20_axis_0_0'] 
data_features_path = ['data_minute/000002/data_features/000002_train_rbm_size_20_axis_0_0', 'data_minute/000002/data_features/000002_val_rbm_size_20_axis_0_0', 'data_minute/000002/data_features/000002_test_rbm_size_20_axis_0_0' ]
data_y_path = ['data_minute/000002/data_features/000002_train_set_y_axis_0_0', 'data_minute/000002/data_features/000002_val_set_y_axis_0_0', 'data_minute/000002/data_features/000002_test_set_y_axis_0_0'] 

PRETRAINING_EPOCH = 100
TRAINING_EPOCH = 3000
RBM_N_HIDDEN = [20]         # number of hidden layer units
RBM_SEQ_LENGTH = 10  # how many steps to unroll
RBM_INPUT_SIZE = RBM_SEQ_LENGTH * 4 
TO_FINTUNING = 1
IS_WRITTING_W = 0
AXIS1 = 0
AXIS2 = 0 # 开盘价， 最高，收盘价，最低
RBM_LEARNING_RATE = 0.01   # learning rate
RBM_BATCH_SIZE = 1
OUTPUT_SIZE = 2
IS_SAVING_LABELS = 1
IS_SAVINT_RBM_FEATURES = 0

SEQ_LENGTH = 25     # how many steps to unroll
N_HIDDEN = 200         # number of hidden layer units
LEARNING_RATE = 0.01   # learning rate
GRAD_CLIP = 100        # 
NUM_EPOCHS = 100 
BATCH_SIZE = 10
INPUT_SIZE = 20
INI = lasagne.init.Uniform(0.1)
max_grad_norm = 10


def test_DBN(finetune_lr=RBM_LEARNING_RATE, pretraining_epochs=PRETRAINING_EPOCH,
             pretrain_lr=0.01, k=1, training_epochs=TRAINING_EPOCH,
             RBM_BATCH_SIZE=RBM_BATCH_SIZE, axis_1=0, axis_2 = 0):
    print('... loading data')
    dataHandle = DataHandle(2,  6)
    train_set_x, train_set_y = dataHandle.load_data(datapath[0], RBM_SEQ_LENGTH, axis_1, axis_2)
    train_data_size = len(train_set_x)

    valid_set_x, valid_set_y = dataHandle.load_data(datapath[1], RBM_SEQ_LENGTH, axis_1, axis_2)
    valid_data_size = len(valid_set_x)

    test_set_x, test_set_y = dataHandle.load_data(datapath[2], RBM_SEQ_LENGTH, axis_1, axis_2)
    test_data_size = len(test_set_x)

    # 存储对应的y类号
    if IS_SAVING_LABELS: 
        with open(data_y_path[0],'wb') as f: 
            for item in train_set_y:
                f.write('%s\n' % item)
        with open(data_y_path[1], 'wb') as f:
            for item in valid_set_y:
                f.write('%s\n' % item)
        with open(data_y_path[2], 'wb') as f:
            for item in test_set_y:
                f.write('%s\n' % item)
    
    # 归一化
    train_set_x = dataHandle.minmax_norm(train_set_x)
    valid_set_x = dataHandle.minmax_norm(valid_set_x)
    test_set_x = dataHandle.minmax_norm(test_set_x)

    # 转化为共享变量
    train_set_x, train_set_y = dataHandle.shared_dataset(train_set_x, train_set_y)
    valid_set_x, valid_set_y = dataHandle.shared_dataset(valid_set_x, valid_set_y)
    test_set_x, test_set_y = dataHandle.shared_dataset(test_set_x, test_set_y)
    datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
   
      
    #计算minibatches的个数
    n_train_batches = train_data_size // RBM_BATCH_SIZE
    n_test_batches = test_data_size // RBM_BATCH_SIZE


    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)

    print('... building the model')
    print('the current hidden layers is ', RBM_N_HIDDEN)
    print('the current epochs is ', TRAINING_EPOCH)
    print('the current axis is ', axis_1, axis_2)
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=RBM_INPUT_SIZE,
            hidden_layers_sizes=RBM_N_HIDDEN,
              n_outs=OUTPUT_SIZE)
    
    # start-snippet-2
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                RBM_BATCH_SIZE=RBM_BATCH_SIZE,
                                                k=k)
    print('... pre-training the model')
    start_time = timeit.default_timer()
    # Pre-train layer-wise
    for i in range(dbn.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost ' % (i, epoch), end=' ')
            print(numpy.mean(c))

    end_time = timeit.default_timer()
    # end-snippet-2
    print('The pretraining code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)
    
    # get the output of the last RBM
    
    if(IS_SAVINT_RBM_FEATURES):
        for i in range(len(data_features_path)):
            features = dbn.extract_features(aim_set_x=datasets[i][0],
                                                        RBM_BATCH_SIZE=RBM_BATCH_SIZE) 
            with open(data_features_path[i], 'wb') as f:
                for item in features:
                    np.savetxt(f, item, fmt='%f')
                    #np.savetxt(f, item, fmt='%.2f')
                print('Done for %s\n' % data_features_path[i])

    ########################
    # FINETUNING THE MODEL #
    ########################
   
   

    # dump the w
    if IS_WRITTING_W:
        with open('rbm_w', 'wb') as f:
            for i in range(dbn.n_layers):
                w = dbn.sigmoid_layers[i].W.get_value()
                #print(type(w))
                #print(w.shape)        
                np.savetxt(f, w)
         

if __name__ == '__main__':
    axis_1, axis_2 = sys.argv[1], sys.argv[2]
    test_DBN(axis_1 = int(axis_1), axis_2=int(axis_2))
