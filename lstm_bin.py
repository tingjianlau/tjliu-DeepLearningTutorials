#########################################################################
# File Name: rnn.py
# Author: james
# mail: zxiaoci@mail.ustc.edu
#########################################################################

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
import lasagne

SEQ_LENGTH = 8      # how many steps to unroll
N_HIDDEN = 200        # number of hidden layer units
LEARNING_RATE = 0.01   # learning rate
GRAD_CLIP = 100        # 
NUM_EPOCHS = 100
BATCH_SIZE = 1
INPUT_SIZE = 4
INI = lasagne.init.Uniform(0.1)
max_grad_norm = 10
interval = 12

#train = 'data/data_600519_train'
#val   = 'data/data_600519_val'
#test  = 'data/data_600519_test'
#train ='data_daily/ifly_train'
#val = 'data_daily/ifly_val'
#test = 'data_daily/ifly_test'
train ='data_daily/000001/train.txt'
val = 'data_daily/000001/val.txt'
test = 'data_daily/000001/test.txt'

axis = (int)(sys.argv[1])

def load_data(dataset):
	print('... loading data')
	data = open(dataset, 'r')
	prices = []
	for line in data:
		prices.append(line.split('\n')[0].split('\t')[1:5])
	#return prices[:0:-1]
	return prices[1::]

def Bollinger(N,train_data,k):
	
	GROUPS = len(train_data) - N +1
	Boll_Bands = np.zeros([GROUPS,3])
	data = [0 for i in range(N)]

	for ptr in range(GROUPS):
		for j in range(N):
			data[j] = float(train_data[ptr+j][2])
		#MA	
		MA = sum(data)/N
		#MD
		MD = math.sqrt(sum([(item - MA)**2 for item in data])/N)
		#MB
		MB = sum(data[0:N-1])/(N-1)
		UP = MB + k*MD
		DN = MB - k*MD
		Boll_Bands[ptr] = [round(item,3) for item in [MA,UP,DN]]
	return Boll_Bands

print("Loading data ...")
train_data = load_data(train)
train_data_size = len(train_data) 
val_data = load_data(val)
val_data_size = len(val_data) 
test_data = load_data(test)
test_data_size = len(test_data) 

def minmax_norm(data):
	max = np.max(data)
	min = np.min(data)
	return (data - min) / (max - min)

def z_score_norm(data, shape=(None, None, None)):
	batch_size = shape[0]
	seq_length = shape[1]
	for j in range(batch_size):
		mean = np.mean(data[j], axis = 0)
		std  = np.std(data[j], axis = 0) 
		data[j] = (data[j] - mean) / std

	return data

def gen_data(p, data, batch_size = BATCH_SIZE, input_size = INPUT_SIZE):
	x = np.zeros((batch_size, SEQ_LENGTH, input_size))
	y = np.zeros((batch_size, 2))
	y_old = np.zeros(batch_size)

	for n in range(batch_size):
		ptr = n
		for i in range(SEQ_LENGTH):
			for j in range(0, input_size):
				x[n, i, j] = data[p+ptr+i][j]
		temp = data[p+ptr+SEQ_LENGTH][axis]
		y_old[n] = data[p+ptr+SEQ_LENGTH-1][axis]
		if ((float)(temp) >= (float)(y_old[n])):
			y[n] = [1, 0]
		else:
			y[n] = [0, 1]
	
	# Center the inputs and outputs
	x = minmax_norm(x)
	#x = z_score_norm(x, (batch_size, SEQ_LENGTH, input_size))
	
	return (x.astype(theano.config.floatX), y.astype(theano.config.floatX),
			y_old.astype(theano.config.floatX))

if __name__ == '__main__':
	print("Building network ...")
	l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE))
	l_forward = lasagne.layers.LSTMLayer(
			l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
			nonlinearity=lasagne.nonlinearities.tanh)
#	l_forward = lasagne.layers.LSTMLayer(
#		l_forward, N_HIDDEN, grad_clipping=GRAD_CLIP,
#		nonlinearity=lasagne.nonlinearities.tanh)
#l_forward = lasagne.layers.LSTMLayer(
#		l_forward, N_HIDDEN, grad_clipping=GRAD_CLIP,
#		nonlinearity=lasagne.nonlinearities.tanh,
#		only_return_final=True)
	l_out = lasagne.layers.DenseLayer(l_forward, num_units=2, 
			nonlinearity=lasagne.nonlinearities.softmax)
	
	target_values = T.matrix('target_output')
	
	network_output = lasagne.layers.get_output(l_out)
	predicted_values = network_output
	cost = lasagne.objectives.categorical_crossentropy(predicted_values, target_values)
	cost = cost.mean()

	test_acc = T.mean(T.eq(T.argmax(predicted_values, axis=1),
		T.argmax(target_values, axis=1)), dtype=theano.config.floatX)

	all_params = lasagne.layers.get_all_params(l_out, trainable=True)

	print("Computing updates ...")
	all_grads = T.grad(cost, all_params)
	all_grads = [T.clip(g, -5, 5) for g in all_grads]
	all_grads, norm = lasagne.updates.total_norm_constraint(
			all_grads, max_grad_norm, return_norm=True)

	sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))

	#updates = lasagne.updates.sgd(all_grads, all_params, learning_rate=sh_lr)
	updates = lasagne.updates.adagrad(cost, all_params, learning_rate=sh_lr)

	print("Compiling functions ...")
	train = theano.function([l_in.input_var, target_values], 
			[cost, predicted_values], updates=updates, allow_input_downcast=True)
	compute_cost = theano.function([l_in.input_var, target_values], 
			[cost, predicted_values, test_acc], allow_input_downcast=True)

	print("Training ...")
	ftrain = open('train_lstm_res.txt', 'wb')
	fval = open('val_lstm_res.txt', 'wb')
	ftest = open('test_lstm_res.txt', 'wb')
	p = 0
	last_cost_val = 10000
	best_cost_val = 10000
	try:
		step = SEQ_LENGTH + BATCH_SIZE - 1
		for it in xrange(NUM_EPOCHS):
			print('epoch %d, lrate = %f' % (it, sh_lr.get_value()))
			for _ in range(train_data_size / BATCH_SIZE):
				x,y,y_mean = gen_data(p, train_data)
				cost, pred = train(x,y)
			#	pred = T.argmax(pred,axis=1)
			#	y = T.argmax(y,axis=1)
				ftrain.write("predicted = {}, actual_value = {}\n"
						.format((pred), (y)))
				# to reuse previous batch, i.e. last batch is data[0:10], next batch 
				# will become data[1:11] instead of data[11:20]
				p += BATCH_SIZE
				if(p + step >= train_data_size):
					p = 0

			pp = 0
			cost_val = 0
			acc_val = 0
			n_iter = (val_data_size - step) / BATCH_SIZE
			for _ in range(n_iter):
				x,y,y_mean = gen_data(pp, val_data)
				cost, pred, acc = compute_cost(x, y)
				fval.write("predicted = {}, actual_value = {}\n"
						.format((pred), (y)))
				cost_val += cost
				acc_val += acc
				pp += BATCH_SIZE
				if(pp + step >= val_data_size):
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
		n_iter = (test_data_size - step) / BATCH_SIZE
		for _ in range(n_iter):
			x,y,y_mean = gen_data(pp, test_data)
			cost, pred, acc = compute_cost(x, y)
			#pred = T.argmax(pred,axis=1)
			#y = T.argmax(y,axis=1)
			ftest.write("predicted = {}, actual_value = {}\n"
					.format((pred), (y)))
			acc_test += acc
			pp += BATCH_SIZE
			if(pp + step >= test_data_size):
				break
		acc_test /= n_iter
		print("test acc = {}".format(acc_test))
	except KeyboardInterrupt:
		pass
