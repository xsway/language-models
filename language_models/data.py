
import os
import torch
from collections import defaultdict
import logging


class Dictionary(object):

    def __init__(self, vocab_path, text_path):
        self.word2idx = {}
        self.idx2word = []
        self.word2freq = defaultdict(int)

        try:
            vocab = open(vocab_path).read()
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            self.idx2word = [w for w in vocab.split()]
            self.vocab_file_exists = True
        except FileNotFoundError:
            logging.info("Vocab file not found, creating new vocab file.")
            self.create_vocab(text_path)
            open(vocab_path, "w").write("\n".join([w for w in self.idx2word]))

    def add_word(self, word):
        self.word2freq[word] += 1
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        #return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def create_vocab(self, path):
        with open(path, 'r') as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.add_word(word)

    def file_to_tensor(self, path):
        """Converts a text file for training or testing to a sequence of indices format
           We assume that training and test data has <eos> symbols """
        assert os.path.exists(path)
        with open(path, 'r') as f:
            ntokens = 0
            for line in f:
                words = line.split()
                ntokens += len(words)

        with open(path, 'r') as f:
            ids = torch.LongTensor(ntokens)
            token = 0
            for line in f:
                words = line.split()
                for word in words:
                    if word in self.word2idx:
                        ids[token] = self.word2idx[word]
                    else:
                        ids[token] = self.word2idx["<unk>"]
                    token += 1

        return ids


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary(os.path.join(path, 'vocab.txt'), os.path.join(path, 'train.txt'))
        self.train = self.dictionary.file_to_tensor(os.path.join(path, 'train.txt'))
        self.valid = self.dictionary.file_to_tensor(os.path.join(path, 'valid.txt'))
        self.test = self.dictionary.file_to_tensor(os.path.join(path, 'test.txt'))




