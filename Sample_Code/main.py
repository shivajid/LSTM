# This is a sample code for char rnn

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # This is to supress CPU warnings
import argparse
import logging
import shutil
import numpy as np
import sys
import json
import codecs




def file_read(args):
    # Read and split data.
    logging.info('Reading data from: %s', args.data_file)
    with codecs.open(args.data_file, 'r', encoding=args.encoding) as f:
        text = f.read()
    return  text


# This is the main function that keeps the parameters
def main():

    parser = argparse.ArgumentParser()

    # Data and vocabulary file
    parser.add_argument('--data_file', type=str, default='data/tiny_shakespeare.txt', help='data file')

    parser.add_argument('--encoding', type=str, default='UTF8', help='encoding')

    # Parameters for saving models.
    parser.add_argument('--output_dir', type=str, default="output", help='outputdir')

    parser.add_argument('-n_save', type=int, default=5, help="how many times to save the model during each epoch")

    parser.add_argument('--max-keep', type= int, default=5, help="how many models to keep")

    # Parameters to configure the neural network.
    parser.add_argument('--hidden_size', type= int, default=128, help='size of RNN hidden state')

    parser.add_argument('--num_layers', type=int, default=2, help='number of layers of RNN')

    parser.add_argument('--num_unrolling', type=int, default=10, help='which models to use')

    parser.add_argument('--model', type=str, default='lstm', help='which model to use (rnn, lstm or gru).')

    # Parameters to control the training.
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')

    parser.add_argument('--batch_size', type=int, default=20, help='mini batch size')

    parser.add_argument('--train_frac', type=float, default=0.9, help='fraction of data used for training.')

    parser.add_argument('--valid_frac', type=float, default=0.05, help='fraction of data used for validation.')

    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate, default to 0 (no dropout).')

    parser.add_argument('--input_dropout', type=float, default=0.0, help=('dropout rate on input layer, default to 0 (no dropout),''and no dropout if using one-hot representation.'))

    # Parameters for gradient descent.
    parser.add_argument('--max_grad_norm', type=float, default=5., help='clip global grad norm')

    parser.add_argument('--learning_rate', type=float, default=2e-3,help='initial learning rate')

    parser.add_argument('--decay_rate', type=float, default=0.95, help='decay rate')

    #Parameters to log to a file
    parser.add_argument('--log_to_file', dest='log_to_file', action='store_true',  help=('whether the experiment log is stored in a file underoutput_dir or printed at stdout.'))

    parser.set_defaults(log_to_file=False)

    parser.add_argument('--progress_freq', type=int, default=100, help=('frequency for progress report in training'))

    parser.add_argument('--verbose', type=int, default=0,help='whether to show progress report in training and evalution.')

    # Parameters to feed in the initial model and current best model.
    parser.add_argument('--init_model', type=str,default='',help=('initial model'))

    parser.add_argument('--best_model', type=str,default='',help=('current best model'))

    parser.add_argument('--best_valid_ppl', type=float, default=np.Inf, help=('current valid perplexity'))

    parser.add_argument('--init_dir', type=str, default='',help='continue from the outputs in the given directory')


    parser.add_argument('--debug', dest='debug', action='store_true', help='show debug information')

    parser.set_defaults(debug=False)

    # Parameters for unittesting the implementation.
    parser.add_argument('--test', dest='test', action='store_true',
                        help=('use the first 1000 character to as data'
                              ' to test the implementation'))
    parser.set_defaults(test=False)


    # Parsing the arguments
    args = parser.parse_args()

    args.save_model = os.path.join(args.output_dir, 'save_model/model')
    args.save_best_model = os.path.join(args.output_dir, 'best_model/model')
    args.tb_log_dir = os.path.join(args.output_dir, 'tensorboard_log/')


    # Create necessary directories.
    if args.init_dir:
        args.output_dir = args.init_dir
    else:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        for paths in [args.save_model, args.save_best_model,
                      args.tb_log_dir]:
            os.makedirs(os.path.dirname(paths))


    # Specify logging config.
    if args.log_to_file:
        args.log_file = os.path.join(args.output_dir, 'experiment_log.txt')
    else:
        args.log_file = 'stdout'

        # Set logging file.
    if args.log_file == 'stdout':
        logging.basicConfig(stream=sys.stdout,
                            format='%(asctime)s %(levelname)s:%(message)s',
                            level=logging.INFO,
                            datefmt='%I:%M:%S')
    else:
        logging.basicConfig(filename=args.log_file,
                            format='%(asctime)s %(levelname)s:%(message)s',
                            level=logging.INFO,
                            datefmt='%I:%M:%S')


    print('=' * 60)
    print('All final and intermediate outputs will be stored in %s/' % args.output_dir)
    print('All information will be logged to %s' % args.log_file)
    print('=' * 60 + '\n')


    if args.debug:
        logging.info('args are:\n%s', args)

    # Prepare parameters.
    if args.init_dir:
        with open(os.path.join(args.init_dir, 'result.json'), 'r') as f:
            result = json.load(f)
        params = result['params']
        args.init_model = result['latest_model']
        best_model = result['best_model']
        best_valid_ppl = result['best_valid_ppl']
        if 'encoding' in result:
            args.encoding = result['encoding']
        else:
            args.encoding = 'utf-8'
        args.vocab_file = os.path.join(args.init_dir, 'vocab.json')
    else:
        params = {'batch_size': args.batch_size,
                  'num_unrollings': args.num_unrollings,
                  'hidden_size': args.hidden_size,
                  'max_grad_norm': args.max_grad_norm,
                  'embedding_size': args.embedding_size,
                  'num_layers': args.num_layers,
                  'learning_rate': args.learning_rate,
                  'model': args.model,
                  'dropout': args.dropout,
                  'input_dropout': args.input_dropout}
        best_model = ''
    logging.info('Parameters are:\n%s\n', json.dumps(params, sort_keys=True, indent=4))

    text = file_read(args)

    if args.test:
        text = text[:1000]

    logging.info('Number of characters: %s', len(text))

    if args.debug:
        n = 10
        logging.info('First %d characters: %s', n, text[:n])

    logging.info('Creating train, valid, test split')
    train_size = int(args.train_frac * len(text))
    valid_size = int(args.valid_frac * len(text))
    test_size = len(text) - train_size - valid_size
    train_text = text[:train_size]
    valid_text = text[train_size:train_size + valid_size]
    test_text = text[train_size + valid_size:]

    if args.vocab_file:
        vocab_index_dict, index_vocab_dict, vocab_size = load_vocab(
            args.vocab_file, args.encoding)





if __name__ == '__main__':
    main()
