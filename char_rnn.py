import cPickle as pickle
import lasagne
'''from lasagne.init import Normal
from lasagne.layers import InputLayer

from lasagne.layers import DenseLayer
from lasagne.layers import get_all_layers
from lasagne.layers import get_all_params
from lasagne.nonlinearities import tanh, softmax'''

import numpy as np


def save_weights(weights, filename):
    with open(filename, 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_weights(layer, filename):
    with open(filename, 'rb') as f:
        src_params_list = pickle.load(f)

    dst_params_list = lasagne.layers.get_all_params(layer)
    # assign the parameter values stored on disk to the model
    for src_params, dst_params in zip(src_params_list, dst_params_list):
        dst_params.set_value(src_params)
        # Load parameters
    '''with np.load(filename) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(layer, param_values)

    # Binarize the weights
    params = lasagne.layers.get_all_params(layer)
    for param in params:
        # print param.name
        if param.name == "W":
            param.set_value(binary_ops.SignNumpy(param.get_value()))'''



def build_model(input_shape, num_hidden, num_output, grad_clipping, rnn, mode):
    
    print('building graph!')
    print('RNN mode:' + rnn)
    print('quantization mode:' + mode)
    print('############################')
    
    if rnn == "LSTM":
        import model.LSTM
        rnn = model.LSTM.LSTMLayer
    elif rnn == "GRU":
        import model.GRU
        rnn = model.GRU.GRULayer
    elif rnn == "Recurrent":
        import model.Recurrent
        rnn = model.Recurrent.RecurrentLayer

    print('building Input Layer')
    l_in = lasagne.layers.InputLayer(input_shape, name='l_in')
    print('---->')

    print('building Fisrt Layer')
    l_rnn1 = rnn(
        l_in, name='l_rnn1',
        num_units=num_hidden, grad_clipping=grad_clipping, mode = mode
    )
    print('---->')

    print('building Second Layer')
    l_rnn2 = rnn(
        l_rnn1, name='l_rnn2',
        num_units=num_hidden, grad_clipping=grad_clipping, mode = mode,
        only_return_final=True,
    )
    print('---->')

    print('building Output Layer')
    l_out = lasagne.layers.DenseLayer(
        l_rnn2, name='l_out', W=lasagne.init.Normal(),
        num_units=num_output, nonlinearity=lasagne.nonlinearities.softmax
    )


    layers = lasagne.layers.get_all_layers(l_out)
    print('building graph done')
    print('############################')
    return {layer.name: layer for layer in layers}


if __name__ == '__main__':
    print('testing build_model')
    build_model((None, 32, 64), 128, 10, 10., 'LSRM', 'normal')
    print('done')
