The codes train Vanilla, LSTM, GRU RNN models with different rounding methods, including Binarization, Ternarization, Pow2-Ternarization and Exponential Quantization.

regularization.py and round_op.py are forked from ottj/QuantizedRNN. generate_samples.py, theano_funcs.py, utils.py and trian_char_rnn.py are forked from hjweide/lasagne-char-rnn. So is char_rnn.py, with some changes.

Training 
Start training model using the command:

$ python train.py

the default parameters include:

rnn type - LSTM

quantization mode - normal

number of hidden state - 128

batch size - 50

you can pass arguments using argparse.

Requirements:
Python 2.7, Numpy, Scipy
Theano
A fast Nvidia GPU (or a large amount of patience)
Setting your Theano flags to use the GPU
Pylearn2
tqdm
sklearn
cPickle 

