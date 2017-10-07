import char_rnn
import theano_funcs
import utils.utils
import argparse
import numpy as np
import lasagne
#from lasagne.layers import get_all_param_values
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_fpath', type=str, default='data/',
                       help='data directory containing input.txt')
    parser.add_argument('--rnn',type=str, default='LSTM',
                        help='tpye of rnn:lstm, gru or recurrent')
    parser.add_argument('--mode',type=str, default = 'normal',
                        help='method to quantize:normal, binary, ternary, dual-copy or quantize')
    parser.add_argument('--weights_fpath',type=str, default = 'cv/',      # the format of weight is .pickle , can be changed to .npz (modify the function load_weight() in char_rnn.py)
                        help='path of weitght')
    parser.add_argument('--num_hidden', type=int, default=128,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')  
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--train_seq_length', type=int, default=50,
                       help='RNN sequence length in training phase')
    parser.add_argument('--grad_clipping', type=float, default=1.,
                       help='clip gradients at this value') 
    parser.add_argument('--load_model', type = bool, default = False,
                        help='whether load a pre-traind model')
    args = parser.parse_args()
    test(args)


def test(args):
    rnn = args.rnn # type of RNN
    mode = args.mode # quantization methods

    weights_fpath = args.weights_fpath  # weights will be stored here
    text_fpath = args.text_fpath  # path to the input file
    grad_clipping = args.grad_clipping  
    num_hidden = args.num_hidden
    batch_size = args.batch_size
    #sample_every = args.sample_every  # sample every n batches
    # sequence length during training, number of chars to draw for sampling
    train_seq_length = args.train_seq_length
    load_model = args.load_model

    
    text_test, vocab_test = utils.utils.parse(text_fpath + 'test.txt')

    #if ((vocab_train == vocab_vali) && (vocab_train == vocab_test)):
    #    print('Vocabulory established')
    # ***ML: need to be modified

    print vocab_test

    # encode each character in the vocabulary as an integer

    encoder = LabelEncoder()
    encoder.fit(list(vocab_test))
    vocab_size = len(vocab_test)

    # ML: build model!
    layers = char_rnn.build_model(
        (None, train_seq_length, vocab_size),  # input_shape
        num_hidden, vocab_size, grad_clipping, rnn, mode
    )

    char_rnn.load_weights(layers['l_out'], weights_fpath + 'weights.pickle')

    print('compiling theano function for testing')
    test_char_rnn = theano_funcs.create_vali_func(layers) # ML:testing flow is as same as validation flow

    try:
    	test_losses = []
        seq_iter_test = utils.utils.sequences(
                text_test, batch_size, train_seq_length, vocab_size, encoder 
        )
        print("Start testing flow:")
        for i, (X_test, y_test) in tqdm(enumerate(seq_iter_test), leave=False):
            if X_test is not None and y_test is not None:
                loss = test_char_rnn(X_test, y_test)
                test_losses.append(loss)
                print(' loss = %.6f' % (loss))
        print("Testing flow finished")
        print('Test set average loss = %.6f'% (np.mean(test_losses)))

    except KeyboardInterrupt:
    	print('caught ctrl-c, stopping training')

if __name__ == '__main__':
	main()