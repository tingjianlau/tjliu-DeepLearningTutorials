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

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from logistic_sgd import LogisticRegression, load_data
from DBN import DBN
from DataHandle import DataHandle

datapath = ['data_daily/000001/data_000001_train', 'data_daily/000001/data_000001_val', 'data_daily/000001/data_000001_test'] 


PRETRAINING_EPOCH = 40
TRAINING_EPOCH = 1000
AXIS1 = 0
AXIS2 = 0 # 开盘价， 最高，收盘价，最低
SEQ_LENGTH = 10  # how many steps to unroll
N_HIDDEN = [20]         # number of hidden layer units
LEARNING_RATE = 0.01   # learning rate
GRAD_CLIP = 100        # 
NUM_EPOCHS = 1000
BATCH_SIZE = 1
OUTPUT_SIZE = 2
INPUT_SIZE = SEQ_LENGTH * 4 


def test_DBN(finetune_lr=0.1, pretraining_epochs=PRETRAINING_EPOCH,
             pretrain_lr=0.01, k=1, training_epochs=TRAINING_EPOCH,
             batch_size=BATCH_SIZE, axis_1=0, axis_2 = 0):
    print('... loading data')
    dataHandle = DataHandle(1,  5)
    train_set_x, train_set_y = dataHandle.load_data(datapath[0], SEQ_LENGTH, axis_1, axis_2)
    train_data_size = len(train_set_x)

    valid_set_x, valid_set_y = dataHandle.load_data(datapath[1], SEQ_LENGTH, axis_1, axis_2)
    valid_data_size = len(valid_set_x)

    test_set_x, test_set_y = dataHandle.load_data(datapath[2], SEQ_LENGTH, axis_1, axis_2)
    test_data_size = len(test_set_x)

    # 归一化
    train_set_x = dataHandle.minmax_norm(train_set_x)
    valid_set_x = dataHandle.minmax_norm(valid_set_x)
    test_set_x = dataHandle.minmax_norm(test_set_x)

    # 转化为共享变量
    train_set_x, train_set_y = dataHandle.shared_dataset(train_set_x, train_set_y)
    valid_set_x, valid_set_y = dataHandle.shared_dataset(valid_set_x, valid_set_y)
    test_set_x, test_set_y = dataHandle.shared_dataset(test_set_x, test_set_y)
    datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

    #train_set_x的访问方式
    #print(train_set_x.get_value(borrow=True))
    
    # test_set_y的访问方式
    #tmp = test_set_y.eval()

    #计算minibatches的个数
    n_train_batches = train_data_size // BATCH_SIZE
    n_valid_batches = valid_data_size// BATCH_SIZE
    n_test_batches = test_data_size // BATCH_SIZE


    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)

    print('... building the model')
    print('the current hidden layers is ', N_HIDDEN)
    print('the current epochs is ', TRAINING_EPOCH)
    print('the current axis is ', axis_1, axis_2)

    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=INPUT_SIZE,
            hidden_layers_sizes=N_HIDDEN,
              n_outs=OUTPUT_SIZE)
    
    # start-snippet-2
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=BATCH_SIZE,
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

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('... finetuning the model')
    # early-stopping parameters

    # look as this many examples regardless
    patience = 10000 
    #patience = numpy.inf 
    #patience = 4 * n_train_batches

    # wait this much longer when a new best is found
    patience_increase = 2.

    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995

    # go through this many minibatches before checking the network on
    # the validation set; in this case we check every epoch
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    #while (epoch < training_epochs):
    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            finetune_cost = train_fn(minibatch_index)
            this_finetune_cost = numpy.mean(finetune_cost)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, finetune cost %.2f, validation error %f %%' % (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_finetune_cost,
                    this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                            improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                          test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete with best validation score of %f %%, '
           'obtained at iteration %i, '
           'with test performance %f %%'
           ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The fine tuning code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)

if __name__ == '__main__':
    axis_1, axis_2 = sys.argv[1], sys.argv[2]
    #test_DBN()
    test_DBN(axis_1 = int(axis_1), axis_2=int(axis_2))
