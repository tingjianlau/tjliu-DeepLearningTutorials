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
from DataHandle import DataHandle, DataHandle_minute
from datetime import datetime

ROOT_PATH = 'data_minute'
STOCK_ID = 'SZ000001'

IS_PRINT_STDOUT = 0 # 是否将输出设置为标准输出
addtion_str = '_including_rbms'
#addtion_str = '_excluding_rbms'
#addtion_str = '_reshape_lstm'

GIBBS_K = 1    
RBMS_MAX_EPOCH = 100
RBM_N_HIDDEN = [5]         # number of hidden layer units
RBM_SEQ_LENGTH = 1  # how many steps to unroll
RBM_INPUT_SIZE = RBM_SEQ_LENGTH * 48 * 4
NETWORK_INPUT_SIZE = RBM_SEQ_LENGTH * 48 * 4
IS_SAVING_RBM_W_HBIAS = False
PRICE_ROW_INTERVAL = [0, 6]
RBM_LEARNING_RATE = 0.01   # learning rate
RBM_BATCH_SIZE = 10
OUTPUT_SIZE = 2
IS_SAVING_LABELS = False
IS_SAVINT_RBM_FEATURES = False 

LSTM_SEQ_LEN = 5     # how many steps to unroll
LSTM_N_HIDDEN = 200         # number of hidden layer units
LSTM_LEARNING_RATE = 0.01   # learning rate
GRAD_CLIP = 100        # 
LSTM_NUM_EPOCHS = 100 
LSTM_BATCH_SIZE = 10
LSTM_INPUT_SIZE = RBM_N_HIDDEN[-1]
INI = lasagne.init.Uniform(0.1)
max_grad_norm = 10


USING_RBMs = 1

def build_lstm(input_var=None, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.)):
    '''
    This creates a LSTM of one hidden layer of 200 units,
    followed by a softmax output layer of 2 units. It applies
    no dropout to the input or hidden layer.
    '''
    l_in = lasagne.layers.InputLayer(shape=(LSTM_BATCH_SIZE*LSTM_SEQ_LEN, NETWORK_INPUT_SIZE), input_var=input_var)

    l_rbms = []
    for i in range(len(RBM_N_HIDDEN)):
        if i==0:
            l_rbms.append(
                    lasagne.layers.DenseLayer(
                        l_in, 
                        num_units=RBM_N_HIDDEN[i],
                        W=W[i],
                        b=b[i],
                        nonlinearity=lasagne.nonlinearities.sigmoid
                        )
                    )
        else:
            l_rbms.append(
                    lasagne.layers.DenseLayer(
                        l_rbms[i-1],
                        num_units=RBM_N_HIDDEN[i],
                        W=W[i],
                        b=b[i],
                        nonlinearity=lasagne.nonlinearities.sigmoid
                        )
                    )
    # reshape不会产生新的W和b参数

    l_reshape = lasagne.layers.ReshapeLayer(l_rbms[-1],
            (LSTM_BATCH_SIZE, LSTM_SEQ_LEN, RBM_N_HIDDEN[-1]))
    #l_reshape = lasagne.layers.ReshapeLayer(l_in,
    #        (LSTM_BATCH_SIZE, LSTM_SEQ_LEN, NETWORK_INPUT_SIZE))

    l_forward = lasagne.layers.LSTMLayer(
            l_reshape, LSTM_N_HIDDEN, grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh,
            only_return_final=True)

    # l_out层会产生一个W和b参数
    l_out = lasagne.layers.DenseLayer(l_forward, num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    print('\n-----------The shape of every layer----------')
    print('\t>input layer: ', l_in.output_shape)
    #for i in range(len(RBM_N_HIDDEN)):
    #    print('\t>rbm layer_{}: {}'.format(i, l_rbms[i].output_shape))
    print('\t>reshape layer: ', l_reshape.output_shape)
    print('\t>lstm layer: ', l_forward.output_shape)
    print('\t>output layer: ', l_out.output_shape)
    print('----------------------End---------------------\n')
    
    return l_out

def gen_data(p, data, data_y, batch_size = LSTM_BATCH_SIZE, input_size = NETWORK_INPUT_SIZE):
    x = np.zeros((batch_size*LSTM_SEQ_LEN, input_size))
    y = np.zeros((batch_size, 2))
    y_old = np.zeros(batch_size)

    for n in range(batch_size):
        ptr = n
        for i in range(LSTM_SEQ_LEN):
            for j in range(0, input_size):
                x[n*LSTM_SEQ_LEN+i][j] = data[p+ptr+i][j]
        
        if (int(data_y[p+ptr+LSTM_SEQ_LEN-1]) == 1):
            y[n] = [1, 0]
        else:
            y[n] = [0, 1]
    
    # Center the inputs and outputs
    x = minmax_norm(x)
    #x = z_score_norm(x, (batch_size, SEQ_LENGTH, input_size))
    
    return (x.astype(theano.config.floatX), y.astype(theano.config.floatX),
            y_old.astype(theano.config.floatX))

def minmax_norm(data):
    max = np.max(data)
    min = np.min(data)
    return (data - min) / (max - min)

# 预训练
def train_RBMs(datasets, batch_size, n_ins, hidden_layers_sizes, k,
                pretrain_epochs, pretrain_lr, data_finetune_features_path,
                data_unfinetune_W_path, data_unfinetune_hbias_path):
    #计算minibatches的个数
    n_train_batches = datasets[0].get_value().shape[0] // batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    
    # construct the RBMs Network
    rbms = RBMs(numpy_rng=numpy_rng, n_ins=n_ins,
            hidden_layers_sizes=hidden_layers_sizes)
    
    # start-snippet-2
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
    pretraining_fns = rbms.pretraining_functions(train_set_x=datasets[0],
                            batch_size=batch_size,k=k)
    print('... pre-training the model')
    start_time = timeit.default_timer()
    # Pre-train layer-wise
    last_cost_mean = 10000
    old_pretrain_lr = pretrain_lr
    for i in range(rbms.n_layers):
        # go through pretraining epochs
        for epoch in range(pretrain_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            cost_mean = abs(numpy.mean(c))
            print('Pre-training layer %i, epoch %d, ls %f, cost %f' 
                    % (i, epoch, pretrain_lr, cost_mean))
            if last_cost_mean <= cost_mean * 1.:
                pretrain_lr /= 2
                if pretrain_lr <= 10e-5:
                    break
            print('Pre-training layer %i, epoch %d, lr %f, cost %f' 
                    % (i, epoch, pretrain_lr, cost_mean))
            last_cost_mean = cost_mean
        pretrain_lr = old_pretrain_lr

    end_time = timeit.default_timer()
    # end-snippet-2
    print('The pretraining code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    rbms_W, rbms_hbias = [], []
    for i in range(rbms.n_layers):
        rbms_W.append(rbms.get_W(i))
        rbms_hbias.append(rbms.get_hbias(i))

    if IS_SAVINT_RBM_FEATURES:
        features = []
        for i in range(len(datasets)):
            features.append(rbms.extract_features(
                    aim_set_x=datasets[i],batch_size=batch_size)) 
        #print(len(features[0][0].shape))
        save_as_txt(features, data_finetune_features_path)

    if IS_SAVING_RBM_W_HBIAS:
        rbms_W_value, rbms_hbias_value = [], []
        for i in range(rbms.n_layers):
            rbms_W_value.append(rbms_W[i].get_value())
            rbms_hbias_value.append(rbms_hbias[i].get_value())
        #print(len(rbms_W_value[0][0].shape))
        save_as_txt(rbms_W_value, data_unfinetune_W_path)
        save_as_txt(rbms_hbias_value, data_unfinetune_hbias_path)

    return rbms_W, rbms_hbias

def save_as_txt(data, topath):
    '''
    data, topath:   list
    一次性写入多个数据集
    '''
    assert len(data)==len(topath)
    for i in range(len(topath)):
        with open(topath[i], 'wb') as f:
            if isinstance(data[i][0], numpy.ndarray):
                # 对于rbm的features(3D)试用
                if len(data[i][0].shape)>1:
                    for item in data[i]:
                        np.savetxt(f, item, fmt='%f')
                else:
                    # 对于rbm的W和hbias(2D)适用
                    np.savetxt(f, data[i], fmt='%f')  
            else:
                for item in data[i]:
                    f.write('%s\n' % item)
            print('Done for writing {}'.format(topath[i]))

def format_path(axes, rbm_n_hidden):
    # 原始数据的路径
    data_x_path = [
        os.path.join(ROOT_PATH, STOCK_ID, 'data_{}_train'.format(STOCK_ID)),
        os.path.join(ROOT_PATH, STOCK_ID, 'data_{}_val'.format(STOCK_ID)),
        os.path.join(ROOT_PATH, STOCK_ID, 'data_{}_test'.format(STOCK_ID))]

    # 保存label的路径
    data_y_path = [
        os.path.join(ROOT_PATH, STOCK_ID, 'data_set_y', 
        '{}_train_set_y_axes_{}_{}'.format(STOCK_ID, axes[0], axes[1])),
        os.path.join(ROOT_PATH, STOCK_ID, 'data_set_y', 
        '{}_val_set_y_axes_{}_{}'.format(STOCK_ID, axes[0], axes[1])),
        os.path.join(ROOT_PATH, STOCK_ID, 'data_set_y', 
        '{}_test_set_y_axes_{}_{}'.format(STOCK_ID, axes[0], axes[1]))]

    size_str = 'size_'
    for i in range(len(rbm_n_hidden)):
        size_str += str(rbm_n_hidden[i])
        size_str += '_'
    sec_folder = 'data_features'

    # 保存RBMs预训练结束之后，即未经微调的特征值的路径
    data_unfinetune_features_path = [
        os.path.join(ROOT_PATH, STOCK_ID, sec_folder, 
            '{}_train_unfinetune_{}axes_{}_{}'.format(STOCK_ID, size_str, axes[0], axes[1])),
        os.path.join(ROOT_PATH, STOCK_ID, sec_folder, 
            '{}_val_unfinetune_{}axes_{}_{}'.format(STOCK_ID, size_str, axes[0], axes[1])),
        os.path.join(ROOT_PATH, STOCK_ID, sec_folder, 
        '{}_test_unfinetune_{}axes_{}_{}'.format(STOCK_ID, size_str, axes[0], axes[1]))]

    # 保存整个网络训练结束之后，即经微调的特征值的路径
    data_finetune_features_path = [
        os.path.join(ROOT_PATH, STOCK_ID, sec_folder, 
        '{}_train_finetune_{}axes_{}_{}'.format(STOCK_ID, size_str, axes[0], axes[1])),
        os.path.join(ROOT_PATH, STOCK_ID, sec_folder, 
        '{}_val_finetune_{}axes_{}_{}'.format(STOCK_ID, size_str, axes[0], axes[1])),
        os.path.join(ROOT_PATH, STOCK_ID, sec_folder, 
        '{}_test_finetune_{}axes_{}_{}'.format(STOCK_ID, size_str, axes[0], axes[1]))]
        
    # 保存RBMs预训练结束之后，即未经微调的W和Bias的路径
    sec_folder = 'data_w_b'
    data_unfinetune_W_path = []
    data_unfinetune_hbias_path = []
    for i in range(len(rbm_n_hidden)):
        data_unfinetune_W_path.append(
                os.path.join(ROOT_PATH, STOCK_ID, sec_folder,
                    '{}_unfinetune_W_hiddenlayer_{}_axes_{}_{}'.format(STOCK_ID,
                        i, axes[0], axes[1])))
        data_unfinetune_hbias_path.append(
                os.path.join(ROOT_PATH, STOCK_ID, sec_folder,
                    '{}_unfinetune_hbias_hiddenlayer_{}_axes_{}_{}'.format(STOCK_ID,
                        i, axes[0], axes[1])))

    # 保存rbm-lstm训练之后的分类结果
    sec_folder = 'data_res'
    data_res_path = []
    data_res_path.append(
            os.path.join(ROOT_PATH, STOCK_ID, sec_folder,
                '{}_res_train_{}seqlen_{}_lstm_{}_seqlen_{}_axes_{}_{}.txt'.format(STOCK_ID, size_str, RBM_SEQ_LENGTH, LSTM_N_HIDDEN, LSTM_SEQ_LEN, axis_1, axis_2)))
    data_res_path.append(
            os.path.join(ROOT_PATH, STOCK_ID, sec_folder,
                '{}_res_val_{}seqlen_{}_lstm_{}_seqlen_{}_axes_{}_{}.txt'.format(STOCK_ID, size_str, RBM_SEQ_LENGTH, LSTM_N_HIDDEN, LSTM_SEQ_LEN, axis_1, axis_2)))
    data_res_path.append(
            os.path.join(ROOT_PATH, STOCK_ID, sec_folder,
                '{}_res_test_{}seqlen_{}_lstm_{}_seqlen_{}_axes_{}_{}.txt'.format(STOCK_ID, size_str, RBM_SEQ_LENGTH, LSTM_N_HIDDEN, LSTM_SEQ_LEN, axis_1, axis_2)))

    return  data_x_path, data_y_path, data_unfinetune_features_path,data_finetune_features_path, data_unfinetune_W_path, data_unfinetune_hbias_path, data_res_path

def test_RBMs_lstm(axes,
        RBMS_MAX_EPOCHs=RBMS_MAX_EPOCH,
        rbm_seq_len=RBM_SEQ_LENGTH,
        pretrain_lr=RBM_LEARNING_RATE, k=1,
        rbm_batch_size=RBM_BATCH_SIZE,
        lstm_batch_size=LSTM_BATCH_SIZE,
        row_interval=PRICE_ROW_INTERVAL):
    '''
    Parameters
    ----------

    '''
    # 格式化路径
    data_x_path, data_y_path, data_unfinetune_features_path,  data_finetune_features_path, data_unfinetune_W_path, data_unfinetune_hbias_path, data_res_path = format_path(axes, RBM_N_HIDDEN)

    # 加载数据
    #dataHandle = DataHandle(RBM_SEQ_LENGTH, PRICE_ROW_INTERVAL, axes, '\t', False)
    dataHandle = DataHandle_minute(RBM_SEQ_LENGTH, PRICE_ROW_INTERVAL, axes, ',', False, 2)
    shared_datasets, unshared_datasets = dataHandle.get_datasets(data_x_path)
    #data_set_x = [unshared_datasets[0][0], unshared_datasets[1][0], unshared_datasets[2][0]]
    #data_set_y = [unshared_datasets[0][1], unshared_datasets[1][1], unshared_datasets[2][1]]
    
    
     # 存储对应的y类号
    if IS_SAVING_LABELS: 
        save_as_txt([unshared_datasets[0][1], unshared_datasets[1][1],
            unshared_datasets[2][1]], data_y_path)
    
    print('\n---------------The Network Info---------------')
    print('\t>time: ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("\t>trainset shape: " , shared_datasets[0].get_value().shape) 
    print("\t>valtset shape: " , shared_datasets[1].get_value().shape) 
    print("\t>testset shape: " , shared_datasets[2].get_value().shape) 
    print('\t>rbms hidden layers: ', RBM_N_HIDDEN)
    print('\t>rbms seq length: ', RBM_SEQ_LENGTH)
    print('\t>axes: ', axes)
    print('\t>lstm hidden layers: ', LSTM_N_HIDDEN)
    print('\t>lstm seq length: ', LSTM_SEQ_LEN)
    print('-----------------------End--------------------\n')
    print('... building the rbms')
    
    # pretraining RBMs and getting the W and hbias of the RBMs
    rbms_W, rbms_hbias = train_RBMs(shared_datasets,
            RBM_BATCH_SIZE, RBM_INPUT_SIZE, RBM_N_HIDDEN,
            GIBBS_K, RBMS_MAX_EPOCH, RBM_LEARNING_RATE, 
            data_finetune_features_path, data_unfinetune_W_path,
            data_unfinetune_hbias_path)
    
    ########################
    # FINETUNING THE MODEL #
    ########################
    # 数据集的大小
    train_data_size = len(unshared_datasets[0][0])
    valid_data_size = len(unshared_datasets[1][0])
    test_data_size = len(unshared_datasets[2][0])

    #print('train_data_size: {}'.format(train_data_size))
    #print('val_data_size: {}'.format(valid_data_size))
    #print('test_data_size: {}'.format(test_data_size))
    
    # prepare theano variables for inputs and targets
    input_var = T.matrix('inputs')
    #input_var = T.tensor3('inputs')
    target_var = T.matrix('targets')
    
    # build the network
    print('\nbuilding network ...')
    #l_out = build_lstm(input_var)
    l_out = build_lstm(input_var, rbms_W, rbms_hbias)
    '''
    all_layers = lasagne.layers.get_all_layers(l_out)
    print('The current network architecture is\n--------------------')
    for item in all_layers:
        print('\t%s' % item)
    '''
    
    # Create loss expression for training
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
    #print(all_params)
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
    #sys.exit(0)
    print('Finishing compiling train function ran for %.2fm' % ((end_time-start_time)/60.))

    # under sampling of data. pos and neg samples are 1:1
   
    # Training the network
    print("Training ...")
    ftrain = open(data_res_path[0], 'wb')
    fval = open(data_res_path[1], 'wb')
    ftest = open(data_res_path[2], 'wb')
    p = 0
    last_cost_val = 10000
    best_cost_val = 10000
    try:
        step = LSTM_SEQ_LEN+ LSTM_BATCH_SIZE- 1
        for it in xrange(LSTM_NUM_EPOCHS):
            print('epoch %d, lrate = %f' % (it, sh_lr.get_value()))
            for _ in range(train_data_size // LSTM_BATCH_SIZE):
                x,y,y_mean = gen_data(p, unshared_datasets[0][0], unshared_datasets[0][1])
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
                x,y,y_mean = gen_data(pp, unshared_datasets[1][0],unshared_datasets[1][1])
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
            x,y,y_mean = gen_data(pp, unshared_datasets[2][0], unshared_datasets[2][1])
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

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print('Pls enter axes')
        sys.exit(0)
    start_running = timeit.default_timer()
    axis_1, axis_2 = int(sys.argv[1]), int(sys.argv[2])

    # 改变输入流
    if IS_PRINT_STDOUT!=1:
        size_str = 'rbms_'
        for i in range(len(RBM_N_HIDDEN)):
            size_str += str(RBM_N_HIDDEN[i])
            size_str += '_'

        log_path = os.path.join(ROOT_PATH, STOCK_ID, 'log', 'rbm-lstm', 
                '{}_log_{}seqlen_{}_lstm_{}_seqlen_{}_axes_{}_{}{}.log'.format(STOCK_ID, size_str, RBM_SEQ_LENGTH, LSTM_N_HIDDEN, LSTM_SEQ_LEN, axis_1, axis_2, addtion_str))
        flog = open(log_path, 'wb')
        old_stdout = sys.stdout
        sys.stdout = flog

    test_RBMs_lstm(axes=[axis_1, axis_2])

    end_running = timeit.default_timer()
    print('\n\nThe network ran for {:.2f}m'.format((end_running - start_running)/60.))

    if IS_PRINT_STDOUT!=1:
        sys.stdout = old_stdout
        print('Done!\nLog path is {}'.format(log_path))
