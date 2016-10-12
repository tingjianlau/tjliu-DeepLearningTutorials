#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

from  LogisticRegression import LogisticRegression

N_EPOCHS = 100
INPUT_SIZE = 40
OUTPUT_SIZE = 2
BATCH_SIZE = 1
LEARNING_RATE = 0.01
N_HIDDEN = [20]
best_model_path = './best_model/best_logistic_for_random_model.pkl'
datapath = ['data/tmp_train.txt', 'data/tmp_val.txt', 'data/tmp_test.txt'] 

def load_data(path):
    data = [[] for i in range(len(path))]
    features = [[] for i in range(len(path))]
    labels = [[] for i in range(len(path))]
    for i in range(len(path)):
        data[i] = open(path[i], 'r')

        for line in data[i]:
            curline = line.split('\n')[0].split()
            features[i].append(curline[0 : INPUT_SIZE])
            labels[i].append(curline[-1])

    #print(features[1])

    def shared_dataset(data_set_x, data_set_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_set_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_set_y, dtype=theano.config.floatX), borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')
    
    #将数据集转换成SharedVariable类型
    rval = []
    for i in range(len(path)):
        set_x, set_y = shared_dataset(features[i], labels[i])
        rval.append((set_x, set_y))

    return rval


def sgd_optimization_mnist(learning_rate=LEARNING_RATE, n_epochs=N_EPOCHS,
                           dataset=datapath,
                           batch_size=BATCH_SIZE):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(datapath)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print(n_train_batches, n_test_batches, n_test_batches)
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=INPUT_SIZE, n_out=OUTPUT_SIZE)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y), classifier.y_pred, classifier.p_y_given_x],
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 500000  # look as this many examples regardless 设置太小可能导致某一次验证误差
                        # 没有之前的好，而过早退出循环
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
                                  # 意思是如果model的this validation error相对于best
                                  # validation error缩小到了99.5%，
                                  # 那么patience就可以提高。所以称为“提高阀值”
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    #while (epoch < 1) and (not done_looping):
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        # 对train_dataset的所有mini_batch进行一遍update
        for minibatch_index in range(n_train_batches):

            # 每个batch都进行GD，并进行参数的update。也就是训练了模型
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            # 经过若干次update迭代后，对验证集进行一次验证，以检测当前model的效果
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                # 取平均的目的是消除valid_batch_size对loss的影响
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                #if 1:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    #test_losses, pred_list = [test_model(i)
                    #               for i in range(n_test_batches)]
                    test_losses = []
                    y_pred= []
                    p_y_given_x = []
                    for i in range(n_test_batches):
                        a, b,c  = test_model(i)
                        test_losses.append(a)
                        y_pred.append(b)
                        p_y_given_x.append(c)
                     
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    #with open(best_model_path, 'wb') as f:
                    #     pickle.dump(classifier, f)

            # 如果很多次update之后（iter不断增加），但是model一直没有很显著的提升（patience没有增大），
            # 那么当update次数超过patience之后，整个训练过程提前结束（early-stopping）
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

    #return pred_list, test_set_y.eval()
    return test_set_y.eval().tolist(), y_pred, p_y_given_x

def predict(best_model, start_index, end_index):
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = pickle.load(open(best_model))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset=datapath
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[start_index:end_index])
    #print("Predicted values for the first 10 examples in test set:")

    return predicted_values


if __name__ == '__main__':
    test_set_y, y_pred, p_y_given_x = sgd_optimization_mnist()
    print(type(test_set_y)) # list
    print(type(y_pred))
    print(type(p_y_given_x))
    print(type(p_y_given_x[0])) # numpy.ndarray

    print(len(y_pred))
    print(len(test_set_y))
    f = open('logistic_sgd_for_random_res.txt', 'wb')
    cnt = 0
    for i in range(len(test_set_y)):
        if test_set_y[i] != y_pred[i][0]:
            cnt += 1
            f.write(str(i) + ' > ' + str(test_set_y[i]) + ' ' + str(y_pred[i][0])+ ' '+ str(p_y_given_x[i][0]) + '\n')

    f.write('Total error: %d\nThe error rate is %f\n' % (cnt, float(cnt)/len(y_pred)))
    f.close()
