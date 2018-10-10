import argparse

lm_parser = argparse.ArgumentParser(add_help=False)
group = lm_parser.add_argument_group('Main parameters')
group.add_argument('--data', type=str, required=True,
                       help='location of the data corpus with train/valid/test split')
group.add_argument('--seed', type=int, default=1111,
                       help='random seed')
group.add_argument('--cuda', action='store_true',
                       help='use CUDA')

model_parser = argparse.ArgumentParser(add_help=False)
group = model_parser.add_argument_group('Model parameters')
group.add_argument('--rnncell', type=str, default='LSTM',
                       help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
group.add_argument('--emsize', type=int, default=200,
                       help='size of word embeddings')
group.add_argument('--nhid', type=int, default=200,
                       help='number of hidden units per layer')
group.add_argument('--nlayers', type=int, default=2,
                       help='number of layers')
group.add_argument('--hidden_sizes', type=int, nargs='+', default=None,
                       help='a list of hidden sizes for RNN layers, '
                            'if specified subsumes and overrides --nhid and --nlayers options')
group.add_argument('--dropout', type=float, default=0.2,
                       help='dropout applied to layers (0 = no dropout)')
group.add_argument('--tied', type=str, default=None,
                       help="add this option to tie the input word embedding matrix and the output "
                            "hidden-to-scores matrix; specify 'standard' or 'plusL' tying")
group.add_argument('--pretrained_embs', type=str, default=None,
                       help='path to the file contraining pre-trained word embeddings')
group.add_argument('--fixed_embs', action='store_true',
                       help="don't tune pre-trained embeddings")
group.add_argument('--mode', type=str, default="forward",
                       help="type of the LM: 'forward' (uses left context), 'backward' (right context) "
                            "or 'bidir' (both contexts but with fixed window defined by bptt parameter)")

train_parser = argparse.ArgumentParser(add_help=False)
group = train_parser.add_argument_group('Training parameters')
group.add_argument('--lr', type=float, default=20,
                       help='initial learning rate')
group.add_argument('--clip', type=float, default=0.25,
                       help='gradient clipping')
group.add_argument('--epochs', type=int, default=40,
                       help='upper epoch limit')
group.add_argument('--batch_size', type=int, default=20, metavar='N',
                       help='batch size')
group.add_argument('--bptt', type=int, default=35,
                       help='backprop through time, that is, effective sequence length during training')
group.add_argument('--save', type=str, default='model.pt',
                       help='path to save the final model')
group.add_argument('--log-interval', type=int, default=200, metavar='N',
                       help='report interval')
group.add_argument('--log', type=str, default='log.txt',
                       help='path to logging file')
group.add_argument('--optimizer', type=str, default="sgd",
                       help='sgd or adam')

