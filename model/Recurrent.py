import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne

from regularization import *
from round_op import *

class Gate(lasagne.layers.Gate):
    """
    This class extends the Lasagne Gate to support rounding of weights
    """
    def __init__(self,mode='normal',H=1.0,nonlinearity=lasagne.nonlinearities.sigmoid,bias_init=lasagne.init.Constant(-1.), **kwargs):
        if mode=='binary':
            if nonlinearity==lasagne.nonlinearities.tanh:
                nonlinearity=binary_tanh_unit
            elif nonlinearity==lasagne.nonlinearities.sigmoid:
                nonlinearity=binary_sigmoid_unit

        super(Gate, self).__init__(nonlinearity=nonlinearity,b=bias_init, **kwargs)


class RecurrentLayer(lasagne.layers.RecurrentLayer):
    """
    This class extends the lasagne RecurrentLayer to support rounding of weights
    """
    def __init__(self, incoming, num_units, 
        stochastic=True, H='glorot', W_LR_scale="Glorot",mode='normal',integer_bits=0,fractional_bits=1, 
                random_seed=666,batch_norm=True,round_hid=True,bn_gamma=lasagne.init.Constant(0.1),mean_substraction_rounding=False,round_bias=True,round_input_weights=True,round_activations=False, **kwargs):

        self.H=H
        self.mode=mode
        self.srng=RandomStreams(random_seed)
        self.stochastic=stochastic
        self.integer_bits=integer_bits
        self.fractional_bits=fractional_bits
        self.batch_norm=batch_norm
        self.round_hid=round_hid
        self.mean_substraction_rounding=mean_substraction_rounding
        self.round_bias=round_bias
        self.round_input_weights=round_input_weights
        self.round_activations=round_activations

        print self.name
        print "Round HID: "+str(self.round_hid)

        if not(mode=='binary' or mode=='ternary' or mode=='dual-copy' or mode=='normal' or mode == 'quantize'):
            raise AssertionError("Unexpected value of 'mode' ! ", mode)

        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            print 'num_inputs: ',num_inputs
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))

        if mode=='binary' or mode=='ternary' or mode=='dual-copy' or mode=='quantize' :
            super(RecurrentLayer, self).__init__(incoming, num_units, **kwargs)
            # add the bianry tag to weights
            if self.round_input_weights:
                self.params[self.W_in_to_hid]=set(['binary'])
            if self.round_hid:
                self.params[self.W_hid_to_hid]=set(['binary'])
            if self.round_bias:
                self.params[self.b]=set(['binary'])

        else :
            super(RecurrentLayer, self).__init__(incoming, num_units, **kwargs)

        self.high = np.float32(np.sqrt(6. / (num_inputs + num_units)))
        self.high_hid = np.float32(np.sqrt(6. / (num_inputs + num_units)))
        self.w0 = np.float32(self.high)
        self.w0_hid = np.float32(self.high_hid)

        input_shape = self.input_shapes[0]
        print 'Input shape: {0}'.format(input_shape)
        #http://github.com/Lasagne/Lasagne/issues/577
        if self.batch_norm:
            print "BatchNorm activated!"
            self.bn = lasagne.layers.BatchNormLayer(input_shape,axes=(0,1),gamma=bn_gamma)
            self.params.update(self.bn.params)
        else:
            print "BatchNorm deactivated!"
        '''
        self.W_in_to_hid_d=T.zeros(self.W_in_to_hid.shape, self.W_in_to_hid.dtype)
        self.W_hid_to_hid_d=T.zeros(self.W_hid_to_hid.shape, self.W_hid_to_hid.dtype)
        self.b_d=T.zeros(self.b.shape, self.b.dtype)    '''


    def get_output_for(self, inputs,deterministic=False, **kwargs):
        if not self.stochastic and not deterministic:
            deterministic=True
        print "deterministic mode: ", deterministic
        def apply_regularization(weights,hid=False):
            current_w0 = self.w0
            if hid:
                current_w0=self.w0_hid

            if self.mean_substraction_rounding:
                return weights
            elif self.mode == 'ternary':

                return ternarize_weights(weights, w0=current_w0, deterministic=deterministic,srng=self.srng)

            elif self.mode == "binary":
                return binarize_weights(weights, 1., self.srng, deterministic=deterministic)
            elif self.mode == "dual-copy":
                return quantize_weights(weights, srng=self.srng, deterministic=deterministic)
            else:
                return weights
        if self.round_input_weights:
            self.Wb_in_to_hid = apply_regularization(self.W_in_to_hid)

        if self.round_hid:
            self.Wb_hid_to_hid = apply_regularization(self.W_hid_to_hid)

        if self.round_bias:
            self.bb = apply_regularization(self.b)


        if self.round_input_weights:
            Wr_in_to_hid = self.W_in_to_hid

        if self.round_hid:
            Wr_hid_to_hid = self.W_hid_to_hid
        if self.round_bias:
            br = self.b

        if self.round_input_weights:
            self.W_in_to_hid = self.Wb_in_to_hid

        if self.round_hid:
            self.W_hid_to_hid = self.Wb_hid_to_hid

        if self.round_bias:
            self.b = self.bb


        input = inputs[0]


        if self.batch_norm:
            input = self.bn.get_output_for(input,deterministic=deterministic, **kwargs)
            if len(inputs) > 1:
                new_inputs=[input,inputs[1]]
            else:
                new_inputs=[input]
        else:
            new_inputs=inputs

        inputs=new_inputs


        input = inputs[0]

        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        if input.ndim > 3:
            input = T.flatten(input, 3)


        input = input.dimshuffle(1,0,2)
        seq_len , num_batch, _ = input.shape


        W_in_stacked = T.concatenate(
            [self.W_in_to_hid], axis=1)

        W_hid_stacked = T.concatenate(
            [self.W_hid_to_hid], axis=1)

        b_stacked = T.concatenate(
            [self.b], axis=0)

        if self.precompute_input:

            input = T.dot(input, W_in_stacked) + b_stacked



        def step(input_n, hid_previous, *args):

            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping,self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:

                input_n = T.dot(input_n, W_in_stacked) + b_stacked


            hid = self.nonlinearity(hid_input + input_n)

            return hid

        def step_masked(input_n, mask_n, hid_previous, *args):
            hid = step(input_n, hid_previous, *args)

            hid = T.switch(mask_n, hid, hid_previous)

            return hid


        if mask is not None:


            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked

        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, lasagne.layers.Layer):

            hid_init = T.dot(T.ones((num_batch,1)), self.hid_init)

        non_seqs = [W_hid_stacked]


        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:

            input_shape = self.input_shapes[0]

            hid_out = lasagne.utils.unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:


            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:

            hid_out = hid_out.dimshuffle(1, 0, 2)

            if self.backwards:
                hid_out = hid_out[:, ::-1]


        if self.round_input_weights:
            self.W_in_to_hid = Wr_in_to_hid

        if self.round_hid:
            self.W_hid_to_hid = Wr_hid_to_hid

        if self.round_bias:
            self.b = br

        return hid_out

def compute_rnn_grads(loss,network):

    layers = lasagne.layers.get_all_layers(network)
    grads = []

    for layer in layers:

        params = layer.get_params(binary=True)
        if params:
            for param in params:
                #print(param.name)
                if param.name == layer.name + '.'+ 'input_to_hidden.W':   # ML: should be Wb?
                    grads.append(theano.grad(loss, wrt=layer.Wb_in_to_hid))
                elif param.name == layer.name + '.'+ 'hidden_to_hidden.W':
                    grads.append(theano.grad(loss, wrt=layer.Wb_hid_to_hid))  
                elif param.name == layer.name + '.'+ 'input_to_hidden.b':
                    grads.append(theano.grad(loss, wrt=layer.bb))
            

    return grads