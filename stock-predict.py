#########################################################################
# File Name: stock-predict.py
# Author: james
# mail: zxiaoci@mail.ustc.edu
#########################################################################
#!/usr/bin/python
from __future__ import print_function

__docformat__ = 'restructedtext en'

#import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
import lasagne

import ConfigParser
import cPickle

n_hidden        = 200
grad_clip       = 100
num_epochs      = 100
no_decay_epoch  = 20
min_improvement = 1.001
input_size      = 6
decay           = 0.5
ini             = lasagne.init.Uniform(0.1)

# read config file to set parameters
cf = ConfigParser.ConfigParser()
cf.read('config.cfg')

# read by type
seq_len     = cf.getint('sec_a'  , 'seq_length')
batch_size  = cf.getint('sec_a'  , 'batch_size')
lr          = cf.getfloat('sec_a', 'learn_rate')
net         = cf.get('sec_a'     , 'net')
binary      = cf.get('sec_a'     , 'binary')
binary      = (binary == 'True')

data        = cf.get('sec_b'     , 'data')
src         = cf.getint('sec_b'  , 'src')
dst         = cf.getint('sec_b'  , 'dst')
cv          = cf.getint('sec_b'  , 'cv')

if len(sys.argv) < 4:
    print('Usage: python stock-prediction.py [stock-num] [src] [dst]')
    sys.exit(-1)

stock_num = sys.argv[1]
data = '/home/james/theano/stock-prediction/data/{}.txt'.format(stock_num)
src  = (int)(sys.argv[2])
dst  = (int)(sys.argv[3])

# load the raw stock data 
def load_data(dataset):
    '''
    Parameters
    ----------
    dataset : str
        Path to input dataset
    
    Returns
    -------
    Returns a numpy array with each row representing 
    one day's data (open/high/close/low/volume/)
    '''
    print('... loading data')
    data = open(dataset, 'r')
    prices = np.zeros((0, input_size))
    for line in data:
        prices = np.append(prices, [line.split('\n')[0].split('\t')[1:1+input_size]], axis=0)
    
    return prices

def reorder(x_in, batch_size, seq_len, cv, dst):
    '''
    Rearrange data set so batches process sequential data.

    Parameters
    ----------
    x_in : 2D numpy.array
    batch_size : int
    seq_len    : int
        number of steps the model is unrolled
    cv         : int
        number of folders in k-fold cross-validation
    dst        : int
        predict axis

    Returns
    -------
    reordered x_in and targets
    '''
    if np.ndim(x_in) != 2:
        raise ValueError("Data must be 2D, was", np.ndim(x_in))

    n_samples = np.shape(x_in)[0] - seq_len + 1
    train_samples = n_samples - 1

    # create targets
    if binary:
        targets = np.zeros((n_samples, 2))
        for i in range(train_samples):
            if (x_in[i+seq_len][dst] >= x_in[i+seq_len-1][src]):
                targets[i] = [1, 0]
            else:
                targets[i] = [0, 1]
        # test target
        targets[-1] = [0, 1]
    else:
        targets = np.zeros((n_samples))
        for i in range(train_samples):
            old = x_in[i+seq_len-1][src]
            new = x_in[i+seq_len][dst]
            targets[i] = (((float)(new) - (float)(old)) * 10.0) / (float)(old)
        # test target
        targets[-1] = 0.0

    # create samples, including test samples
    x_in = [x_in[i:i+seq_len] for i in range(n_samples)]

    # list -> array
    x_in = np.array(x_in).astype(theano.config.floatX)
    targets = np.array(targets).astype(theano.config.floatX)

    # auxiliary function. min-max normalization
    def minmax_norm(data):
        max = np.max(data)
        min = np.min(data)

        # for numerical stability 
        if max == min :
            data = 0.5
            return data

        return (data - min) / (max - min)

    # auxiliary function. z_core normlization
    def z_score_norm(data, shape=(None, None, None)):
        batch_size = shape[0]
        seq_length = shape[1]
        for j in range(batch_size):
            mean = np.mean(data[j], axis = 0)
            std  = np.std(data[j], axis = 0) 
            data[j] = (data[j] - mean) / std

        return data

    # batch normalization
    for i in range(n_samples):
        for j in range(input_size):
            x_in[i][:,j] = minmax_norm(x_in[i][:,j])

    # k-fold cross validation
    def split_dataset(data, target, cv):
        arr = np.arange(train_samples)
        np.random.shuffle(arr) # generate random seq
        fold_size = train_samples // cv
        x_out = []; x_split = []
        t_out = []; t_split = []
        cnt_num = 0; cnt_split = 0
        for i in range(train_samples):
            x_split.append(data[arr[i]])
            t_split.append(target[arr[i]])
            cnt_num += 1
            if cnt_num >= fold_size:
                cnt_split += 1
                x_array = np.array(x_split)
                x_out.append(x_array)
                t_array = np.array(t_split)
                t_out.append(t_array)
                x_split = []; t_split = []
                cnt_num = 0

        return (np.array(x_out), np.array(t_out))

    #train_data = np.tensor4(split_dataset(x_in, targets, cv))
    train_data = split_dataset(x_in, targets, cv)
    test_data = (x_in[-1], targets[-1])

    return (train_data, test_data)

def traindata(seq_len, batch_size):
    x = load_data(data)
    return reorder(x, batch_size, seq_len, cv, dst)

def build_lstm(input_var=None):
    '''
    This creates an LSTM of one hidden layer of 200 units,
    followed by a softmax output layer of 2 units. It applies
    no dropout to the input or hidden layer
    '''
    l_in = lasagne.layers.InputLayer(shape=(batch_size, seq_len, input_size), input_var=input_var)
    l_forward = lasagne.layers.LSTMLayer(
            l_in, n_hidden, grad_clipping=grad_clip,
            nonlinearity=lasagne.nonlinearities.tanh,
            only_return_final=True)
    
    if binary:
        l_out = lasagne.layers.DenseLayer(l_forward, num_units=2, 
            nonlinearity=lasagne.nonlinearities.softmax)
    else:
        l_out = lasagne.layers.DenseLayer(l_forward, num_units=1, 
            nonlinearity=lasagne.nonlinearities.tanh)

    return l_out

train_data, test_data = traindata(seq_len, batch_size)
x_data, y_data = train_data
x_test, y_test = test_data
#x_data, y_data, x_test, y_test = traindata(seq_len, batch_size)

if __name__ == '__main__':
    # Prepare theano variables for inputs and targets   
    input_var = T.tensor3('inputs')
    if binary:
        target_var = T.matrix('targets')
    else:
        target_var = T.vector('targets')

    # Build the network
    print("Building network ...")
    l_out = build_lstm(input_var)
    
    # Create loss expression for training
    print(lasagne.layers.get_output_shape(l_out))   
    network_output = lasagne.layers.get_output(l_out)
    if binary:
        predicted_values = network_output
        cost = lasagne.objectives.categorical_crossentropy(predicted_values, target_var).mean()
    else:
        predicted_values = network_output.flatten()
        cost = T.mean((predicted_values - target_var)**2)

    # Compute testing accuracy
    if binary:
        test_acc = T.mean(T.eq(T.argmax(predicted_values, axis=1),
        T.argmax(target_var, axis=1)), dtype=theano.config.floatX)
    else:
        test_acc = T.mean(T.eq(T.argmax(predicted_values),
        T.argmax(target_var)), dtype=theano.config.floatX)

    # Update the trainable parameters
    print("Computing updates ...")
    sh_lr = theano.shared(lasagne.utils.floatX(lr))
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)
    updates = lasagne.updates.adagrad(cost, all_params, learning_rate=sh_lr)

    # Compile training and validating functions
    print("Compiling functions ...")
    train = theano.function([input_var, target_var], 
            [cost, predicted_values], updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([input_var, target_var], 
            [cost, predicted_values, test_acc], allow_input_downcast=True)

    # under sampling of data. pos and neg samples are 1:1
    def under_sample(data, target):
        n_samples = data.shape[0]
        n_pos = 0; n_neg = 0; pos_idx = []; neg_idx = []
        if binary:
            for i in range(n_samples):
                if target[i] == [0, 1]:
                    n_neg += 1; neg_idx.append(i)
                else:
                    n_pos += 1; pos_idx.append(i)
        else:
            for i in range(n_samples):
                if target[i] < 0.0:
                    n_neg += 1; neg_idx.append(i)
                else:
                    n_pos += 1; pos_idx.append(i)
        if n_neg > n_pos:
            np.random.shuffle(neg_idx)
            neg_idx = neg_idx[0:n_pos]
            for j in range(n_pos):
                neg_idx.append(data[pos_idx[j]])
            np.random.shuffle(neg_idx)
            return data[neg_idx]
        else:
            np.random.shuffle(pos_idx)
            pos_idx = pos_idx[0:n_neg]
            for j in range(n_neg):
                pos_idx.append(data[neg_idx[j]])
            np.random.shuffle(pos_idx)
            return data[pos_idx]

    # Training the network
    ftrain = open('train.txt', 'wb')
    fval = open('val.txt', 'wb')
    fout = open('out.txt', 'a')
    print("Training ...")
    last_cost_val = 10000
    for it in range(num_epochs):
        print ('epoch %d, lr = %f' % (it, sh_lr.get_value()))
        all_cost_val = 0
        for tst_idx in range(cv):
            x_train = np.concatenate((x_data[:tst_idx], x_data[tst_idx+1:]))
            y_train = np.concatenate((y_data[:tst_idx], y_data[tst_idx+1:]))
            x_valid = x_data[tst_idx]
            y_valid = y_data[tst_idx]
        
            n_train_samples = x_train.shape[0] * x_train.shape[1]
            x_train = x_train.reshape(n_train_samples, seq_len, input_size)
            if binary:
                y_train = y_train.reshape(n_train_samples, 2)
            else:
                y_train = y_train.reshape(n_train_samples)
            n_train_batches = n_train_samples // batch_size
        
            n_valid_samples = x_valid.shape[0]
            n_valid_batches  = n_valid_samples // batch_size
        
            # training
            for i in range(n_train_batches):
                x_batch = x_train[i*batch_size:(i+1)*batch_size]
                y_batch = y_train[i*batch_size:(i+1)*batch_size]
                cost, pred = train(x_batch, y_batch)
                if binary:
                    ftrain.write("predicted = {}, actual_value = {}\n"
                        .format((pred), (y_batch)))
                else:
                    y_old = x_batch[:,-1,src]
                    ftrain.write("predicted = {}, actual_value = {}\n"
                        .format((pred/10+1)*y_old, (y_batch/10+1)*y_old))
        
            # validating
            cost_val = 0; acc_val = 0
            for i in range(n_valid_batches):
                x_batch = x_valid[i*batch_size:(i+1)*batch_size]
                y_batch = y_valid[i*batch_size:(i+1)*batch_size]
                cost, pred, acc = compute_cost(x_batch, y_batch)
                if binary:
                    fval.write("predicted = {}, actual_value = {}\n"
                        .format((pred), (y_batch)))
                else:
                    y_old = x_batch[:,-1,src]
                    fval.write("predicted = {}, actual_value = {}\n"
                        .format((pred/10+1)*y_old, (y_batch/10+1)*y_old))
                cost_val += cost; acc_val += acc
            cost_val /= (n_valid_batches)
            acc_val  /= (n_valid_batches)
            print("cost = {}, acc = {}".format(cost_val, acc_val))
            all_cost_val += cost_val

        # halve lrate if performance decays
        if (last_cost_val <= all_cost_val*min_improvement and it >= no_decay_epoch):
            lasagne.layers.set_all_param_values(l_out, all_param_values)
            current_lr = sh_lr.get_value()
            sh_lr.set_value(lasagne.utils.floatX(current_lr * float(decay)))
            if (sh_lr.get_value() <= tol):
                break
        else:
            all_param_values = lasagne.layers.get_all_param_values(l_out)
        last_cost_val = all_cost_val
        
    # save weights
    #all_param_values = lasagne.layers.get_all_param_values(l_out)
    model_name = '/home/james/theano/stock-prediction/model/{}_{}_{}.model.pickle'.format(stock_num, src, dst)
    with open(model_name, 'wb') as f:
        cPickle.dump(all_param_values, f, cPickle.HIGHEST_PROTOCOL)
        
    # testing on the last day
    print("Testing ...")
    _, pred, _ = compute_cost([x_test], [y_test])
    result = ['up', 'down']
    idx = np.argmax(pred[0])
    #if (src == 0):
    fout.write('-------------------\nstock number %s\n-------------------\n' % (stock_num))
    fout.write('in(%d), out(%d), pred(%s), prob(%f)\n' % (src, dst, result[idx], pred[0][idx]))
