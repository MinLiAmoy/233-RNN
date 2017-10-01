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
    #parser.add_argument('--log_dir', type=str, default='logs',
    #                  help='directory containing tensorboard logs')
    parser.add_argument('--rnn',type=str, default='LSTM',
                        help='tpye of rnn:lstm, gru or recurrent')
    parser.add_argument('--mode',type=str, default = 'normal',
                        help='method to quantize:normal, binary, ternary, dual-copy or quantize')
    #parser.add_argument('--save_dir', type=str, default='save',
    #                   help='directory to store checkpointed models')
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
    parser.add_argument('--sample_seq_length', type=int, default=200,
                       help='RNN sequence length in sampling phase')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--sample_every', type=int, default=1000,
                       help='sample frequency')
    parser.add_argument('--grad_clipping', type=float, default=1.,
                       help='clip gradients at this value') 
    parser.add_argument('--lr', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.97,
                       help='the decay rate of learning rate')
    parser.add_argument('--lr_decay_after', type=int, default =10,
                       help='number of epochs to start decaying the learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop') 
    parser.add_argument('--load_model', type = bool, default = False,
                        help='whether load a pre-traind model')
    args = parser.parse_args()
    train(args)


def train(args):
    rnn = args.rnn # type of RNN
    mode = args.mode # quantization methods

    weights_fpath = args.weights_fpath  # weights will be stored here
    text_fpath = args.text_fpath  # path to the input file
    max_epochs = args.max_epochs
    lr = args.lr
    lr_decay = args.lr_decay
    lr_decay_after = args.lr_decay_after
    grad_clipping = args.grad_clipping  
    num_hidden = args.num_hidden
    batch_size = args.batch_size
    sample_every = args.sample_every  # sample every n batches
    # sequence length during training, number of chars to draw for sampling
    train_seq_length, sample_seq_length = args.train_seq_length, args.sample_seq_length
    load_model = args.load_model

    text, vocab_train = utils.utils.parse(text_fpath+'train.txt')
    text_vali, vocab_vali = utils.utils.parse(text_fpath + 'vali.txt')
    text_test, vocab_test = utils.utils.parse(text_fpath + 'test.txt')

    #if ((vocab_train == vocab_vali) && (vocab_train == vocab_test)):
    #    print('Vocabulory established')
    # ***ML: need to be modified
    print vocab_train
    print vocab_vali
    print vocab_test

    # encode each character in the vocabulary as an integer

    encoder = LabelEncoder()
    encoder.fit(list(vocab_train))
    vocab_size = len(vocab_train)

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
    phrases = ['First Citizen:']

    print('compiling theano function for training')
    train_char_rnn = theano_funcs.create_train_func(layers, rnn)
    print('theano function for training built')

    print('compiling theano function for sampling')
    sample = theano_funcs.create_sample_func(layers)
    print('theano funciton for sampling built')

    print('compiling theano function for validation')
    vali_char_rnn = theano_funcs.create_vali_func(layers)
    print('theano function for validation built')

    print('compiling theano function for testing')
    test_char_rnn = theano_funcs.create_vali_func(layers) # ML:testing flow is as same as validation flow

    best_loss = 10
    best_epoch = 1

    print('Start Training')
    try:
        for epoch in range(1, 1 + max_epochs):
            print('epoch %d' % (epoch))
            if epoch >= lr_decay_after:
                lr = lr * lr_decay
            # sample from the model and update the weights
            train_losses = []
            vali_losses = []
            seq_iter = utils.utils.sequences(
                text, batch_size, train_seq_length, vocab_size, encoder
            )
            for i, (X, y) in tqdm(enumerate(seq_iter), leave=False):
                if X is not None and y is not None:
                    loss = train_char_rnn(X, y, lr)
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

            # ML: start validation flow
            seq_iter_vali = utils.utils.sequences(
                text_vali, batch_size, train_seq_length, vocab_size, encoder 
            )
            print("Start validation flow:")
            for i, (X_vali, y_vali) in tqdm(enumerate(seq_iter_vali), leave=False):
                if X_vali is not None and y_vali is not None:
                    loss = vali_char_rnn(X_vali, y_vali)
                    vali_losses.append(loss)
                    print(' loss = %.6f' % (loss))
            print("Validation flow finished")
            print('Validation loss = %.6f'% (np.mean(vali_losses)))

        # ML: start testing flow
        test_losses = []
        seq_iter_test = utils.utils.sequences(
                text_test, batch_size, train_seq_length, vocab_size, encoder 
        )
        print("Start testing flow:")
        for i, (X_test, y_test) in tqdm(enumerate(seq_iter_test), leave=False):
            if X_test is not None and y_test is not None:
                loss = test_char_rnn(X_test, y_test)
                vali_losses.append(loss)
                print(' loss = %.6f' % (loss))
        print("Testing flow finished")
        print('Test set average loss = %.6f'% (np.mean(test_losses)))




    except KeyboardInterrupt:
        print('caught ctrl-c, stopping training')

    # write the weights to disk so we can try out the model as pickle format
    print('saving weights to %s' % (weights_fpath + 'weights.pickle'))
    weights = lasagne.layers.get_all_param_values(layers['l_out'])  # 'l_out!'
    char_rnn.save_weights(weights, weights_fpath+ 'weights.pickle')
    print('done')


if __name__ == '__main__':
    main()
