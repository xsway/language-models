
import torch
import torch.nn as nn

from rnn_models import RNNLanguageModel, BidirectionalLanguageModel
from utils import batchify, get_batch, update_hidden

from torch.optim.adam import Adam
from simple_sgd import SimpleSGD

import math
import time
import argparse
import data
import lm_argparser

criterion = nn.CrossEntropyLoss()
corpus = None


def train(model, train_data, ntokens, seq_len, args, epoch, optimizer):
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - seq_len, seq_len)):
        data, targets = get_batch(train_data, i, seq_len, args.mode, evaluation=False)

        hidden = update_hidden(model, args.mode, hidden, args.batch_size)
        optimizer.zero_grad()

        output, hidden = model(data, hidden)

        loss = nn.CrossEntropyLoss()(output.view(-1, ntokens), targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            if args.optimizer == "sgd":
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // seq_len, optimizer.lr,
                                  elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            else:
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // seq_len,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(model, data_source, ntokens, seq_len, batch_size, mode):
    model.eval()
    total_loss = 0
    n = 0
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - seq_len, seq_len):
        hidden = update_hidden(model, mode, hidden, batch_size)
        data, targets = get_batch(data_source, i, seq_len, mode, evaluation=True)

        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(targets) * criterion(output_flat, targets).data.item()
        n += len(targets)

    return total_loss / n

def main():

    parser = argparse.ArgumentParser(parents=[lm_argparser.lm_parser,
                                              lm_argparser.model_parser,
                                              lm_argparser.train_parser],
                                     #formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     description="Training LMs")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    print("***** Arguments *****")
    print(args)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    global corpus
    corpus = data.Corpus(args.data)

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size, args.cuda)
    val_data = batchify(corpus.valid, eval_batch_size, args.cuda)
    test_data = batchify(corpus.test, eval_batch_size, args.cuda)

    ntokens = len(corpus.dictionary)
    print("Vocab size", ntokens)

    if not args.hidden_sizes:
        args.hidden_sizes = args.nlayers * [args.nhid, ]

    if args.mode == "bidir":   # bidirectional with fixed window, forward and backward are trained simultaneously
        model = BidirectionalLanguageModel(corpus.dictionary.word2idx, args.emsize, args.hidden_sizes, args.dropout,
                                           args.rnncell, args.pretrained_embs, args.fixed_embs, args.tied)
    else:    # simple forward or backward LMs
        model = RNNLanguageModel(corpus.dictionary.word2idx, args.emsize, args.hidden_sizes, args.dropout,
                                 args.rnncell, args.pretrained_embs, args.fixed_embs, args.tied)

    if args.cuda:
        model.cuda()

    print("********* Model **********")
    print(model)

    print("Number of parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

    print("******** Training ********")
    lr = args.lr
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "adam":
        optimizer = Adam(params, lr=lr, amsgrad=True)
    else:
        optimizer = SimpleSGD(params, lr=lr)

    best_val_loss = None

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            seq_len = args.bptt

            train(model, train_data, len(corpus.dictionary), seq_len, args, epoch, optimizer)
            with torch.no_grad():
                val_loss = evaluate(model, val_data, len(corpus.dictionary), seq_len, eval_batch_size, args.mode)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                if args.optimizer == "sgd":
                    optimizer.update()

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    with torch.no_grad():
        test_loss = evaluate(model, test_data, len(corpus.dictionary), args.bptt, eval_batch_size, args.mode)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

if __name__ == "__main__":
    main()


