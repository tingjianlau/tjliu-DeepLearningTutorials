#-*-coding: utf-8 -*-
"""
RBMs类
"""
from __future__ import print_function, division
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM


# start-snippet-1
'''
我们定义RBMs类，它包含了MLP的层，用于连接RBMs。
因为使用RBMs初始化MLP，所以要尽量降低两个类的耦合度：
RBMs用于初始化网络，MLP用于分类。
'''
class RBMs(object):
    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500]):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the RBMs

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data

        # the data is presented as rasterized images
        self.x = T.matrix('x')

        # the labels are presented as 1D vector of [int] labels
        self.y = T.ivector('y')
        # end-snippet-1
        # The RBMs is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the RBMs as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the RBMs by doing stochastic gradient descent on the
        # MLP.

        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the RBMs if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
            
            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            # self.sigmoid_layers存储了正向图并组成了MLP
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the RBMs. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the RBMs.
            # sigmoid层的参数是RBMs的参数，但RBM层中的hbias不是
            # 即在后续的微调时，因为每个RBM和sigmoid层共享权重和隐层偏置
            # 所以在微调时更新sigmoid层的W和b相当于微调RBM层的W和hbias
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            # 此时RBM的W, b都是来自sigmoid层
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,  # 将sigmoid层的共享变量W作为RBMs层的W，
                                                # 所以sigmoid层和RBMs层共享此变量
                            hbias=sigmoid_layer.b)  # 同理对于RBMs层的hbias
            # self.rbm_layers存储了RBMs并预训练MLP的每一层
            self.rbm_layers.append(rbm_layer)

    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        # 为了改变训练时的学习率，这里使用Theano变量类型，并赋予一个初始值。
        learning_rate = T.scalar('lr')  # learning rate to use

        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.In(learning_rate, value=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    # 当第二层RBM调用fn时，此时的输入应该是上一层RBM的输出
                    # 而不是原始的数据集，不能被givens所蒙骗
                    # 可能是因为第二层RBM的更新中不会直接用到self.x
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    # add by tjliu
    def extract_features(self, aim_set_x, batch_size):
        # compute number of minibatches for training, validation and testing
        n_aim_batches = aim_set_x.get_value(borrow=True).shape[0]
        n_aim_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

         # extract the feature
        extract_features_i = theano.function(
            [index],
            self.sigmoid_layers[-1].output,
            givens={
                self.x: aim_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )      
        return [extract_features_i(i) for i in range(n_aim_batches)]

    def get_W(self, sigmoid_layers_index=0):
        # sigmoid_layers_index: the index of the self.sigmoid_layers
        # return: ndarray
        return self.sigmoid_layers[sigmoid_layers_index].W
    
    def get_hbias(self, sigmoid_layers_index=0):
        return self.sigmoid_layers[sigmoid_layers_index].b
