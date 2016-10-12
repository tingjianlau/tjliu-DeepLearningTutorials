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

datapath = ['data/tmp_train.txt', 'data/tmp_val.txt', 'data/tmp_test.txt'] 
#train ='data/ifly_train'
#val = 'data/ifly_val'
#test = 'data/ifly_test'

SEQ_LENGTH = 0  # how many steps to unroll
N_HIDDEN = 20         # number of hidden layer units
LEARNING_RATE = 0.01   # learning rate
GRAD_CLIP = 100        # 
NUM_EPOCHS = 100
BATCH_SIZE = 1
INPUT_SIZE = 40 
def my_load_data(path):
    data = open(path, 'r')
    prices = []
    label = []
    
    for line in data:
        val = line.split('\n')[0].split()
        prices.append(val[0:40])
        label.append(val[-1])

    return prices, label


def data_prehandle(data_set_x, SEQ_LENGTH, axis = 0):
    data_set_y = []
    new_len = len(data_set_x)-SEQ_LENGTH
    for i in range(new_len):
        if data_set_x[i+SEQ_LENGTH-1][axis] > data_set_x[i+SEQ_LENGTH][axis]:
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

def test_DBN(finetune_lr=0.1, pretraining_epochs=40,
             pretrain_lr=0.01, k=1, training_epochs=100,
             batch_size=BATCH_SIZE):
    # 加载数据
    train_set_x, train_set_y = my_load_data(datapath[0])
    train_data_size = len(train_set_x)

    valid_set_x, valid_set_y = my_load_data(datapath[1])
    valid_data_size = len(valid_set_x)

    test_set_x, test_set_y = my_load_data(datapath[2])
    test_data_size = len(test_set_x)
    #for i in range(len(test_set_x)):
    #    for j in range(len(test_set_x[0])):
    #        fval.write('%s ' % str(test_set_x[i][j]))
    #    fval.write(str('%s \n' % test_set_y[i]))
    #fval.close()

    # 归一化

    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)
    datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    #train_set_x的访问方式
    #print(train_set_x.get_value(borrow=True))
    ftmp = open('tmp.txt', 'wb')
    tmp = test_set_y.eval()
    print(tmp.size)
    np.savetxt(ftmp, tmp, fmt='%10.5f')
    ftmp.close()
    
    # test_set_y的访问方式
    #tmp = test_set_y.eval()
    #for i in range(tmp.size):
    #    ftmp.write(str(tmp[i]))
    #ftmp.close()

    #计算minibatches的个数
    n_train_batches = train_data_size // BATCH_SIZE


    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)

    print('... building the model')
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=INPUT_SIZE,
            hidden_layers_sizes=[20],
              n_outs=2)
    
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
    patience = 4 * n_train_batches

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

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
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
    test_DBN()
