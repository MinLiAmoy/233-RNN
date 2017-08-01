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
    parser.add_argument('--text_fpath', type=str, default='data/parsed.txt',
                       help='data directory containing input.txt')
    #parser.add_argument('--log_dir', type=str, default='logs',
    #                  help='directory containing tensorboard logs')
    parser.add_argument('--rnn',type=str, default='LSTM',
                        help='tpye of rnn:lstm, gru or recurrent')
    parser.add_argument('--mode',type=str, default = 'normal',
                        help='method to quantize:normal, binary, ternary, dual-copy or quantize')
    #parser.add_argument('--save_dir', type=str, default='save',
    #                   help='directory to store checkpointed models')
    parser.add_argument('--weights_fpath',type=str, default = 'cv/',
                        help='method to quantize:normal, binary, ternary, dual-copy or quantize')
    parser.add_argument('--num_hidden', type=int, default=128,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')  # ML: still need to modify
    '''parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')'''
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--train_seq_length', type=int, default=50,
                       help='RNN sequence length in training phase')
    parser.add_argument('--sample_seq_length', type=int, default=200,
                       help='RNN sequence length in sampling phase')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--sample_every', type=int, default=1000,
                       help='sample frequency')
    parser.add_argument('--grad_clipping', type=float, default=1.,
                       help='clip gradients at this value') # ML: need to modify
    parser.add_argument('--lr', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop') # ML: need to modify
    parser.add_argument('--load_model', type = bool, default = False,
                        help='whether load a pre-traind model')
    #parser.add_argument('--gpu_mem', type=float, default=0.666,
    #                   help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
    '''parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)'''
    args = parser.parse_args()
    train(args)


def train(args):
    rnn = args.rnn # type of RNN
    mode = args.mode

    weights_fpath = args.weights_fpath  # weights will be stored here
    text_fpath = args.text_fpath  # path to the input file
    max_epochs = args.max_epochs
    lr = args.lr
    grad_clipping = args.grad_clipping  # ML: need to be modified
    num_hidden = args.num_hidden
    batch_size = args.batch_size
    sample_every = args.sample_every  # sample every n batches
    # sequence length during training, number of chars to draw for sampling
    train_seq_length, sample_seq_length = args.train_seq_length, args.sample_seq_length
    load_model = args.load_model

    text, vocab = utils.utils.parse(text_fpath)

    # encode each character in the vocabulary as an integer
    encoder = LabelEncoder()
    encoder.fit(list(vocab))
    vocab_size = len(vocab)

    # ML: build model!
    layers = char_rnn.build_model(
        (None, train_seq_length, vocab_size),  # input_shape
        num_hidden, vocab_size, grad_clipping, rnn, mode
    )

    # optionally load a pre-trained model
    if load_model:
        print('loading model weights from %s' % (weights_fpath + 'weights.pickle'))
        char_rnn.load_weights(layers['l_out'], weights_fpath + 'weights.pickle')

    # phrases to use during sampling
    #phrases = ['I should go to bed now']
    phrases = [' ']

    print('compiling theano function for training')
    train_char_rnn = theano_funcs.create_train_func(layers, rnn, lr=lr)
    print('theano function for training built')

    print('compiling theano function for sampling')
    sample = theano_funcs.create_sample_func(layers)
    print('theano funciton for sampling built')
    best_loss = 10
    best_epoch = 1

    print('Start Training')
    try:
        for epoch in range(1, 1 + max_epochs):
            print('epoch %d' % (epoch))

            # sample from the model and update the weights
            train_losses = []
            seq_iter = utils.utils.sequences(
                text, batch_size, train_seq_length, vocab_size, encoder
            )
            for i, (X, y) in tqdm(enumerate(seq_iter), leave=False):
                if X is not None and y is not None:
                    loss = train_char_rnn(X, y)
                    train_losses.append(loss)
                    print(' loss = %.6f' % (loss))

                # continuously sample from the model
                if ((i + 1) % sample_every) == 0:
                    print(' loss = %.6f' % (np.mean(train_losses)))
                    phrase = np.random.choice(phrases)
                    generated_phrase = utils.utils.sample(
                        sample, phrase,
                        train_seq_length, sample_seq_length,
                        vocab_size, encoder
                    )
                    print('%s%s' % (phrase, generated_phrase))
            if np.mean(train_losses) < best_loss:
                print ('saving weight to %s in npz format' % (weights_fpath))
                np.savez(weights_fpath + 'rnn_paremeter.npz', *lasagne.layers.get_all_param_values(layers['l_out']))
                best_loss = np.mean(train_losses)
                best_epoch = epoch
            print("  LR:                            "+str(lr))
            print("  training loss:                 "+str(np.mean(train_losses)))
            print("  best epoch:                    "+str(best_epoch))
            print("  best training loss:            "+str(best_loss))


    except KeyboardInterrupt:
        print('caught ctrl-c, stopping training')

    # write the weights to disk so we can try out the model as pickle format
    print('saving weights to %s' % (weights_fpath + 'weights.pickle'))
    weights = lasagne.layers.get_all_param_values(layers['l_out'])  # 'l_out!'
    char_rnn.save_weights(weights, weights_fpath+ 'weights.pickle')
    print('done')


if __name__ == '__main__':
    main()
