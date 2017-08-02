__author__="Joachim Ott"
# -*- coding: utf-8 -*-

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



class LSTMLayer(lasagne.layers.LSTMLayer):
    """
    This class extends the lasagne LSTMLayer to support rounding of weights
    """
    def __init__(self, incoming, num_units, 
        stochastic = True, H='glorot',W_LR_scale="Glorot",mode='nomal',integer_bits=0,fractional_bits=1,
                random_seed=666,batch_norm=True,round_hid=True,bn_gamma=lasagne.init.Constant(0.1),mean_substraction_rounding=False,round_bias=True,round_input_weights=True,round_activations=False, **kwargs):

        self.H=H
        self.mode=mode
        self.srng = RandomStreams(random_seed)
        self.stochastic=stochastic
        self.integer_bits=integer_bits
        self.fractional_bits=fractional_bits
        self.batch_norm=batch_norm
        self.round_hid=round_hid
        self.mean_substraction_rounding=mean_substraction_rounding
        self.round_bias=round_bias
        self.round_input_weights=round_input_weights
        self.round_activations=round_activations
        
        #print self.name
        #print "Round HID: "+str(self.round_hid)

        if not(mode=='binary' or mode=='ternary' or mode=='dual-copy' or mode=='normal' or mode=='quantize'):
            raise AssertionError("Unexpected value of 'mode' ! ", mode)

        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            print 'num_inputs: ',num_inputs
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))

        if mode=='binary' or mode=='ternary' or mode=='dual-copy' or mode=='quantize':
            super(LSTMLayer, self).__init__(incoming, num_units, peepholes = False, **kwargs)
            #add the binary tag to weights
            if self.round_input_weights:
                self.params[self.W_in_to_ingate]=set(['binary'])
                self.params[self.W_in_to_forgetgate]=set(['binary'])
                self.params[self.W_in_to_cell]=set(['binary'])
                self.params[self.W_in_to_outgate]=set(['binary'])
            if self.round_hid:
                self.params[self.W_hid_to_ingate]=set(['binary'])
                self.params[self.W_hid_to_forgetgate]=set(['binary'])
                self.params[self.W_hid_to_cell]=set(['binary'])
                self.params[self.W_hid_to_outgate]=set(['binary'])
            if self.round_bias:
                self.params[self.b_ingate] = set(['binary'])
                self.params[self.b_forgetgate] = set(['binary'])
                self.params[self.b_cell] = set(['binary'])
                self.params[self.b_outgate] = set(['binary'])

        else:
            super(LSTMLayer, self).__init__(incoming, num_units, peepholes = False, **kwargs)

        self.high = np.float32(np.sqrt(6. / (num_inputs + num_units)))
        self.high_hid = np.float32(np.sqrt(6. / (num_units + num_units)))
        self.W0 = np.float32(self.high)
        self.W0_hid = np.float32(self.high_hid)


        input_shape = self.input_shapes[0]
        print 'Input Shape: {0}'.format(input_shape)
        #https://github.com/Lasagne/Lasagne/issues/577
        if self.batch_norm:
            print "BatchNorm activated!"
            self.bn = lasagne.layers.BatchNormLayer(input_shape, axes=(0,1),gamma=bn_gamma)
            self.params.update(self.bn.params)
        else:
            print "BatchNorm deactivated!"

        # ML: maybe it's useless.
        '''self.W_hid_to_ingate_d = T.zeros(self.W_hid_to_ingate.shape, self.W_hid_to_ingate.dtype)
        self.W_hid_to_forgetgate_d = T.zeros(self.W_hid_to_forgetgate.shape, self.W_hid_to_forgetgate.dtype)
        self.W_hid_to_cell_d = T.zeros(self.W_hid_to_cell.shape, self.W_hid_to_cell.dtype)
        self.W_hid_to_outgate_d = T.zeros(self.W_hid_to_outgate.shape, self.W_hid_to_outgate.dtype)
        self.W_in_to_ingate_d = T.zeros(self.W_in_to_ingate.shape, self.W_in_to_ingate.dtype)
        self.W_in_to_forgetgate_d = T.zeros(self.W_in_to_forgetgate.shape, self.W_in_to_forgetgate.dtype)
        self.W_in_to_cell_d = T.zeros(self.W_in_to_cell.shape, self.W_in_to_cell.dtype)
        self.W_in_to_outgate_d = T.zeros(self.W_in_to_outgate.shape, self.W_in_to_outgate.dtype)
        self.b_ingate_d = T.zeros(self.b_ingate.shape, self.b_ingate.dtype)
        self.b_forgetgate_d = T.zeros(self.b_forgetgate.shape, self.b_forgetgate.dtype)
        self.b_cell_d = T.zeros(self.b_cell.shape, self.b_cell.dtype)
        self.b_outgate_d = T.zeros(self.b_outgate.shape, self.b_outgate.dtype)'''


    def get_output_for(self, inputs, deterministic=False, **kwargs):
        if not self.stochastic and not deterministic:
            deterministic=True
        #print "deterministic mode: ",deterministic
        def apply_regularization(weights,hid=False):
            current_W0 = self.W0
            if hid:
                current_W0 = self.W0_hid

            if self.mean_substraction_rounding:
                return weights
            elif self.mode == "ternary":

                return ternarize_weights(weights, W0=current_W0, deterministic=deterministic,
                                        srng=self.srng)
            elif self.mode == "binary":
                return binarize_weights(weights, 1., self.srng, deterministic=deterministic)
            elif self.mode == "dual-copy":
                return dual_copy_rounding(weights, self.integer_bits, self.fractional_bits)
            elif self.mode == "quantize":
                return quantize_weights(weights, srng=self.srng, deterministic=deterministic)
            else:
                return weights

        if self.round_input_weights:
            self.Wb_in_to_ingate = apply_regularization(self.W_in_to_ingate)
            self.Wb_in_to_forgetgate = apply_regularization(self.W_in_to_forgetgate)
            self.Wb_in_to_cell = apply_regularization(self.W_in_to_cell)
            self.Wb_in_to_outgate = apply_regularization(self.W_in_to_outgate)

        if self.round_hid:
            self.Wb_hid_to_ingate = apply_regularization(self.W_hid_to_ingate)
            self.Wb_hid_to_forgetgate = apply_regularization(self.W_hid_to_forgetgate)
            self.Wb_hid_to_cell = apply_regularization(self.W_hid_to_cell)
            self.Wb_hid_to_outgate = apply_regularization(self.W_hid_to_outgate)

        if self.round_bias:
            self.bb_ingate = apply_regularization(self.b_ingate)
            self.bb_forgetgate = apply_regularization(self.b_forgetgate)
            self.bb_cell = apply_regularization(self.b_cell)
            self.bb_outgate = apply_regularization(self.b_outgate)

        if self.round_input_weights:
        #Backup high precision values
            Wr_in_to_ingate = self.W_in_to_ingate
            Wr_in_to_forgetgate = self.W_in_to_forgetgate
            Wr_in_to_cell = self.W_in_to_cell
            Wr_in_to_outgate = self.W_in_to_outgate

        if self.round_hid:
            Wr_hid_to_ingate = self.W_hid_to_ingate
            Wr_hid_to_forgetgate = self.W_hid_to_forgetgate
            Wr_hid_to_cell = self.W_hid_to_cell
            Wr_hid_to_outgate = self.W_hid_to_outgate

        if self.round_bias:   # ML: delete "self."
            br_ingate = self.b_ingate
            br_forgetgate = self.b_forgetgate
            br_cell = self.b_cell
            br_outgate = self.b_outgate

        #Overwrite weights with binarized weights
        if self.round_input_weights:
            self.W_in_to_ingate = self.Wb_in_to_ingate
            self.W_in_to_forgetgate = self.Wb_in_to_forgetgate
            self.W_in_to_cell = self.Wb_in_to_cell
            self.W_in_to_outgate = self.Wb_in_to_outgate

        if self.round_hid:
            self.W_hid_to_ingate = self.Wb_hid_to_ingate
            self.W_hid_to_forgetgate = self.Wb_hid_to_forgetgate
            self.W_hid_to_cell = self.Wb_hid_to_cell
            self.W_hid_to_outgate = self.Wb_hid_to_outgate

        if self.round_bias:
            self.b_ingate = self.bb_ingate
            self.b_forgetgate = self.bb_forgetgate
            self.b_cell = self.bb_cell
            self.b_outgate = self.bb_outgate


        #Retrieve the layer input
        input = inputs[0]

        #Apply BN
        #https://github.com/Lasagne/Lasagne/issues/577
        if self.batch_norm:
            input = self.bn.get_output_for(input,deterministic=deterministic, **kwargs)
            if len(inputs) > 1:
                new_inputs=[input,inputs[1]]
            else:
                new_inputs=[input]
        else:
            new_inputs=inputs

        inputs=new_inputs

        #Retrieve the layer input
        input = inputs[0]
        #Retrieve the mask when it is supplied
        mask = None 
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        #Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to 
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4 * num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(    # ML: delete '_d'
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
            self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
            self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack gate biases into a (4 * num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
            self.b_cell, self.b_outgate], axis=0)

        if self.precompute_input:

            input = T.dot(input, W_in_stacked) + b_stacked

        # at each call to scan , input_n will be (n_time-steps, 4*num_units).
        # we define a slicing funcitno that extract the input to each LSTM gate
        def slice_w(x,n):
            s = x[:, n * self.num_units:(n + 1) * self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)
            return s

        
        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, hid_previous, cell_previous, *arg):
            # compute W_{hi} h_{t-1}, W_{hf} h_{t-1}, W_{hc} h_{t-1}, W_{ho} h_{t-1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xi}x_t + b_{i}, W_{xf}x_t + b_{f}, W_{xc}x_t + b_{c}, W_{xo}x_t + b{o}
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # input, forget and output gates
            ingate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            forgetgate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            outgate = slice_w(hid_input, 3) + slice_w(input_n, 3)
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            outgate = self.nonlinearity_outgate(outgate)

            # compute new cell state
            cell_input = slice_w(hid_input, 2) + slice_w(input_n, 2)
            cell_input = self.nonlinearity_cell(cell_input)
            cell = forgetgate * cell_previous + ingate * cell_input


            # compute o_t emul nonlinearity(c_t)
            hid = outgate * self.nonlinearity(cell)

            return [cell, hid]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            # skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:



            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, lasagne.layers.Layer):

            hid_init = T.dot(T.ones((num_batch,1)), self.hid_init)

        if not isinstance(self.cell_init, lasagne.layers.Layer):

            cell_init = T.dot(T.ones((num_batch,1)), self.cell_init)

        non_seqs = [W_hid_stacked]

        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:

            input_shape = self.input_shapes[0]

            cell_out, hid_out = lasagne.utils.unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])

        else: 
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[cell_init, hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:

            hid_out = hid_out.dimshuffle(1,0,2)

            if self.backwards:
                hid_out = hid_out[:, ::-1]

        #copy back high precision values
        if self.round_input_weights:
            self.W_in_to_ingate = Wr_in_to_ingate
            self.W_in_to_forgetgate = Wr_in_to_forgetgate
            self.W_in_to_cell = Wr_in_to_cell
            self.W_in_to_outgate = Wr_in_to_outgate

        if self.round_hid:
            self.W_hid_to_ingate = Wr_hid_to_ingate
            self.W_hid_to_forgetgate = Wr_hid_to_forgetgate
            self.W_hid_to_cell = Wr_hid_to_cell
            self.W_hid_to_outgate = Wr_hid_to_outgate

        if self.round_bias:  # ML: delete 'self.'
            self.b_ingate = br_ingate
            self.b_forgetgate = br_forgetgate
            self.b_cell = br_cell
            self.b_outgate = br_outgate

        return hid_out

# This function computes the gradient of the binary weights
def compute_rnn_grads(loss,network):

    layers = lasagne.layers.get_all_layers(network)
    grads = []

    for layer in layers:

        params = layer.get_params(binary=True)
        if params:
            for param in params:
                #print(param.name)
                if param.name == layer.name + '.'+ 'W_in_to_ingate':   # ML: should be Wb?
                    grads.append(theano.grad(loss, wrt=layer.Wb_in_to_ingate))
                elif param.name == layer.name + '.'+ 'W_in_to_forgetgate':
                    grads.append(theano.grad(loss, wrt=layer.Wb_in_to_forgetgate))
                elif param.name == layer.name + '.'+ 'W_in_to_cell':
                    grads.append(theano.grad(loss, wrt=layer.Wb_in_to_cell))
                elif param.name == layer.name + '.'+ 'W_in_to_outgate':
                    grads.append(theano.grad(loss, wrt=layer.Wb_in_to_outgate))
                elif param.name == layer.name + '.'+ 'W_hid_to_ingate':
                    grads.append(theano.grad(loss, wrt=layer.Wb_hid_to_ingate))
                elif param.name == layer.name + '.'+ 'W_hid_to_forgetgate':
                    grads.append(theano.grad(loss, wrt=layer.Wb_hid_to_forgetgate))
                elif param.name == layer.name + '.'+ 'W_hid_to_cell':
                    grads.append(theano.grad(loss, wrt=layer.Wb_hid_to_cell))
                elif param.name == layer.name + '.'+ 'W_hid_to_outgate':
                    grads.append(theano.grad(loss, wrt=layer.Wb_hid_to_outgate))    
                elif param.name == layer.name + '.'+ 'b_ingate':
                    grads.append(theano.grad(loss, wrt=layer.bb_ingate))
                elif param.name == layer.name + '.'+ 'b_forgetgate':
                    grads.append(theano.grad(loss, wrt=layer.bb_forgetgate))
                elif param.name == layer.name + '.'+ 'b_cell':
                    grads.append(theano.grad(loss, wrt=layer.bb_cell))
                elif param.name == layer.name + '.'+ 'b_outgate':
                    grads.append(theano.grad(loss, wrt=layer.bb_outgate))

    return grads


