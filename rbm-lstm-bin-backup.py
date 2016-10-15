#!/usr/bin/env python 
# -*- coding: UTF-8 -*-

##########################################################################
#   > File Name: my_rbms.py
#   > Author: Tingjian Lau
#   > Mail: tjliu@mail.ustc.edu.cn
#   > Created Time: 2016/09/20
#   > Detail: 
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
import lasagne

from logistic_sgd import LogisticRegression, load_data
from RBMs import RBMs
from DataHandle_backup import DataHandle

datapath = ['data_daily/000001/data_000001_train', 'data_daily/000001/data_000001_val', 'data_daily/000001/data_000001_test'] 
data_advanced_features_path = ['data_daily/000001/data_features/000001_train_finetune_epoch_3000_size_20_axis_0_0', 'data_daily/000001/data_features/000001_val_finetune_epoch_3000_size_20_axis_0_0', 'data_daily/000001/data_features/000001_test_finetune_epoch_3000_size_20_axis_0_0'] 
data_features_path = ['data_daily/000001/data_features/000001_train_rbm_size_20_axis_0_0', 'data_daily/000001/data_features/000001_val_rbm_size_20_axis_0_0', 'data_daily/000001/data_features/000001_test_rbm_size_20_axis_0_0' ]
data_y_path = ['data_daily/000001/data_features/000001_train_set_y_axis_0_0', 'data_daily/000001/data_features/000001_val_set_y_axis_0_0', 'data_daily/000001/data_features/000001_test_set_y_axis_0_0'] 

PRETRAINING_EPOCH = 4
TRAINING_EPOCH = 3000
RBM_N_HIDDEN = [20]         # number of hidden layer units
RBM_SEQ_LENGTH = 10  # how many steps to unroll
RBM_INPUT_SIZE = RBM_SEQ_LENGTH * 4 
NETWORK_INPUT_SIZE = RBM_SEQ_LENGTH * 4 
TO_FINTUNING = 1
IS_WRITTING_W = 0
AXIS1 = 0
AXIS2 = 0 # 开盘价， 最高，收盘价，最低
RBM_LEARNING_RATE = 0.01   # learning rate
RBM_BATCH_SIZE = 10
OUTPUT_SIZE = 2
IS_SAVING_LABELS = 0
IS_SAVINT_RBM_FEATURES = 0

LSTM_SEQ_LEN = 1     # how many steps to unroll
LSTM_N_HIDDEN = 200         # number of hidden layer units
LSTM_LEARNING_RATE = 0.01   # learning rate
GRAD_CLIP = 100        # 
LSTM_NUM_EPOCHS = 100 
LSTM_BATCH_SIZE = 10
LSTM_INPUT_SIZE = RBM_N_HIDDEN[-1]
INI = lasagne.init.Uniform(0.1)
max_grad_norm = 10

READ_DATA_FROM=1
READ_DATA_TO=5
USING_RBMs = 1

#def build_lstm(input_var=None):
def build_lstm(input_var=None, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.)):
    '''
    This creates a LSTM of one hidden layer of 200 units,
    followed by a softmax output layer of 2 units. It applies
    no dropout to the input or hidden layer.
    '''
    l_in = lasagne.layers.InputLayer(shape=(LSTM_BATCH_SIZE, LSTM_SEQ_LEN, NETWORK_INPUT_SIZE), input_var=input_var)
    #l_in = lasagne.layers.InputLayer(shape=(LSTM_BATCH_SIZE*LSTM_SEQ_LEN, NETWORK_INPUT_SIZE), input_var=input_var)
    print('the shape of input is ', lasagne.layers.get_output_shape(l_in))
    '''
    l_sigmoid = []
    for i in range(len(RBM_N_HIDDEN)):
        if i==0:
            l_sigmoid.append(
                    lasagne.layers.DenseLayer(
                        l_in, 
                        num_units=RBM_N_HIDDEN[i],
                        W=W[i],
                        b=b[i],
                        nonlinearity=lasagne.nonlinearities.sigmoid
                        )
                    )
        else:
            l_sigmoid.append(
                    lasagne.layers.DenseLayer(
                        l_sigmoid[i-1],
                        num_units=RBM_N_HIDDEN[i],
                        W=W[i],
                        b=b[i],
                        nonlinearity=lasagne.nonlinearities.sigmoid
                        )
                    )
        '''
        #print('the shape of sigmoid is ', l_sigmoid[i].output_shape)
        #print('the shape of W is', W[i].get_value().shape)

    #l_reshape = lasagne.layers.ReshapeLayer(l_sigmoid[-1],
    #l_reshape = lasagne.layers.ReshapeLayer(l_in,
    #        (LSTM_BATCH_SIZE, LSTM_SEQ_LEN, 40))
            #(LSTM_BATCH_SIZE, LSTM_SEQ_LEN, LSTM_INPUT_SIZE))
    #print('the shape of reshape is ', l_reshape.output_shape)

    l_forward = lasagne.layers.LSTMLayer(
            #l_reshape, LSTM_N_HIDDEN, grad_clipping=GRAD_CLIP,
            l_in, LSTM_N_HIDDEN, grad_clipping=GRAD_CLIP,
            #l_sigmoid[-1], LSTM_N_HIDDEN, grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh,
            only_return_final=True)
    print('the shape of lstm is ', l_forward.output_shape)

    l_out = lasagne.layers.DenseLayer(l_forward, num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

def gen_data(p, data, data_y, batch_size = LSTM_BATCH_SIZE, input_size = NETWORK_INPUT_SIZE):
    x = np.zeros((batch_size*LSTM_SEQ_LEN, input_size))
    y = np.zeros((batch_size, 2))
    y_old = np.zeros(batch_size)

    for n in range(batch_size):
        ptr = n
        for i in range(LSTM_SEQ_LEN):
            for j in range(0, input_size):
                #print(p+ptr+j)
                x[n*LSTM_SEQ_LEN+i][j] = data[p+ptr+i][j]
        
        if (int(data_y[p+ptr+LSTM_SEQ_LEN-1]) == 1):
            y[n] = [1, 0]
        else:
            y[n] = [0, 1]
    
    # Center the inputs and outputs
    #x = minmax_norm(x)
    #x = z_score_norm(x, (batch_size, SEQ_LENGTH, input_size))
    
    return (x.astype(theano.config.floatX), y.astype(theano.config.floatX),
            y_old.astype(theano.config.floatX))

def test_RBMs_lstm(pretraining_epochs=PRETRAINING_EPOCH,
        pretrain_lr=RBM_LEARNING_RATE, k=1,
        rbm_batch_size=RBM_BATCH_SIZE,
        lstm_batch_size=LSTM_BATCH_SIZE,
        read_data_from=READ_DATA_FROM, read_data_to=READ_DATA_TO,
        axis_1=0, axis_2=0):
    '''
    Parameters
    ----------

    '''
    # 加载数据
    dataHandle = DataHandle(read_data_from, read_data_to)
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
    train_set_x_list, train_set_y_list = train_set_x, train_set_y
    valid_set_x_list, valid_set_y_list = valid_set_x, valid_set_y
    test_set_x_list, test_set_y_list = test_set_x, test_set_y

    # 转化为共享变量
    train_set_x, train_set_y = dataHandle.shared_dataset(train_set_x, train_set_y)
    valid_set_x, valid_set_y = dataHandle.shared_dataset(valid_set_x, valid_set_y)
    test_set_x, test_set_y = dataHandle.shared_dataset(test_set_x, test_set_y)
    datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

    print("the trainset shape is " , datasets[0][0].get_value().shape) 
    #计算minibatches的个数
    n_train_batches = train_data_size // RBM_BATCH_SIZE
    n_test_batches = test_data_size // RBM_BATCH_SIZE


    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)

    print('... building the model')
    print('the current hidden layers is ', RBM_N_HIDDEN)
    print('the current epochs is ', TRAINING_EPOCH)
    print('the current axis is ', axis_1, axis_2)
    print('the current lstm_seq_lenght is ', LSTM_SEQ_LEN)
    # construct the Deep Belief Network
    rbms = RBMs(numpy_rng=numpy_rng, n_ins=RBM_INPUT_SIZE,
            hidden_layers_sizes=RBM_N_HIDDEN,
              n_outs=OUTPUT_SIZE)
    
    # start-snippet-2
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = rbms.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=RBM_BATCH_SIZE,
                                                k=k)
    print('... pre-training the model')
    start_time = timeit.default_timer()
    # Pre-train layer-wise
    for i in range(rbms.n_layers):
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
    
    # get the W and hbias of the RBMs
    print("Getting the W and hbias of the RBMs ...")
    rbms_W = []
    rbms_hbias = []
    for i in range(rbms.n_layers):
        rbms_W.append(rbms.get_W(i))
        rbms_hbias.append(rbms.get_hbias(i))
    
    if(IS_SAVINT_RBM_FEATURES):
        for i in range(len(data_features_path)):
            features = rbms.extract_features(aim_set_x=datasets[i][0],
                                                        RBM_BATCH_SIZE=RBM_BATCH_SIZE) 
            with open(data_features_path[i], 'wb') as f:
                for item in features:
                    np.savetxt(f, item, fmt='%f')
                    #np.savetxt(f, item, fmt='%.2f')
                print('Done for %s\n' % data_features_path[i])

    ########################
    # FINETUNING THE MODEL #
    ########################
    # prepare theano variables for inputs and targets
    if(USING_RBMs):
        input_var = T.matrix('inputs')
    else:
        input_var = T.tensor3('inputs')
    target_var = T.matrix('targets')
    
    # build the network
    print('building network ...')
    #l_out = build_lstm(input_var)
    l_out = build_lstm(input_var, rbms_W, rbms_hbias)
    
    # Create loss expression for training
    print("the shape of network output is ", lasagne.layers.get_output_shape(l_out))
    network_output = lasagne.layers.get_output(l_out)
    predicted_values = network_output
    cost = lasagne.objectives.categorical_crossentropy(
                    predicted_values, target_var).mean()

    # Comput testing accuracy
    test_acc = T.mean(T.eq(T.argmax(predicted_values, axis=1),
        T.argmax(target_var, axis=1)), dtype=theano.config.floatX)

    # Update the trainable parameters
    print('Compute updates ...')
    sh_lr = theano.shared(lasagne.utils.floatX(LSTM_LEARNING_RATE))
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)
    print(all_params)
    updates = lasagne.updates.adagrad(cost, all_params, learning_rate=sh_lr)

    # Compile training and validating functions
    print("Compiling functions ...")
    start_time = timeit.default_timer()
    train = theano.function([input_var, target_var],
            [cost, predicted_values], updates=updates,
            allow_input_downcast=True)
    compute_cost = theano.function([input_var, target_var],
            [cost, predicted_values, test_acc],
            allow_input_downcast=True)
    end_time = timeit.default_timer()
    print('Finishing compiling train function ran for %.2fm' % ((end_time-start_time)/60.))

    # under sampling of data. pos and neg samples are 1:1
   
    # Training the network
    print("Training ...")
    ftrain = open('train_lstm_res.txt', 'wb')
    fval = open('val_lstm_res.txt', 'wb')
    ftest = open('test_lstm_res.txt', 'wb')
    p = 0
    last_cost_val = 10000
    best_cost_val = 10000
    try:
        step = LSTM_SEQ_LEN+ LSTM_BATCH_SIZE- 1
        for it in xrange(LSTM_NUM_EPOCHS):
            print('epoch %d, lrate = %f' % (it, sh_lr.get_value()))
            for _ in range(train_data_size // LSTM_BATCH_SIZE):
                x,y,y_mean = gen_data(p, train_set_x_list, train_set_y_list)
                cost, pred = train(x,y)
                ftrain.write("predicted = {}, actual_value = {}\n"
                        .format((pred), (y)))
                # to reuse previous batch, i.e. last batch is data[0:10], next batch 
                # will become data[1:11] instead of data[11:20]
                p += LSTM_BATCH_SIZE
                if(p + step >= train_data_size):
                    p = 0

            pp = 0
            cost_val = 0
            acc_val = 0
            n_iter = (valid_data_size - step) // LSTM_BATCH_SIZE 
            for _ in range(n_iter):
                x,y,y_mean = gen_data(pp, valid_set_x_list,valid_set_y_list)
                cost, pred, acc = compute_cost(x, y)
                fval.write("predicted = {}, actual_value = {}\n"
                        .format((pred), (y)))
                cost_val += cost
                acc_val += acc
                pp += LSTM_BATCH_SIZE
                if(pp + step >= valid_data_size):
                    break
            cost_val /= n_iter
            acc_val /= n_iter
            print("cost = {}, acc = {}".format(cost_val, acc_val))

            # halve lrate
            if (last_cost_val <= cost_val * 1.001):
                lasagne.layers.set_all_param_values(l_out, all_param_values)
                current_lr = sh_lr.get_value()
                sh_lr.set_value(lasagne.utils.floatX(current_lr / 2))
                if (sh_lr.get_value() <= 10e-5):
                    break
            else:
                all_param_values = lasagne.layers.get_all_param_values(l_out)
                best_cost_val = cost_val
            last_cost_val = cost_val

        lasagne.layers.set_all_param_values(l_out, all_param_values)
        pp = 0
        acc_test = 0
        n_iter = (test_data_size - step) // LSTM_BATCH_SIZE
        for _ in range(n_iter):
            x,y,y_mean = gen_data(pp, test_set_x_list, test_set_y_list)
            cost, pred, acc = compute_cost(x, y)
            #pred = T.argmax(pred,axis=1)
            #y = T.argmax(y,axis=1)
            ftest.write("predicted = {}, actual_value = {}\n"
                    .format((pred), (y)))
            acc_test += acc
            pp += LSTM_BATCH_SIZE
            if(pp + step >= test_data_size):
                break
        acc_test /= n_iter
        print("test acc = {}".format(acc_test))
    except KeyboardInterrupt:
        pass
    # dump the w
    if IS_WRITTING_W:
        with open('rbm_w', 'wb') as f:
            for i in range(rbms.n_layers):
                w = rbms.get_W(i).get_value()
                np.savetxt(f, w)
        with open('rbm_b', 'wb') as f:
            for i in range(rbms.n_layers):
                b = rbms.get_hbias(i).get_value()
                np.savetxt(f, b)
         

if __name__ == '__main__':
    axis_1, axis_2 = sys.argv[1], sys.argv[2]
    test_RBMs_lstm(axis_1 = int(axis_1), axis_2=int(axis_2))
