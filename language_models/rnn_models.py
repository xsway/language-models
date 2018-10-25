
import torch.nn as nn
from modules import Encoder, Decoder, StackedRNN
import utils


class RNNLanguageModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, word2idx, emb_size, hidden_sizes, dropout,
                 rnn_type="LSTM", pretrained_embs=None, fixed_embs=False, tied=None):
        super(RNNLanguageModel, self).__init__()

        self.encoder = Encoder(word2idx, emb_size, pretrained_embs, fixed_embs)
        self.decoder = Decoder(len(word2idx), hidden_sizes[-1], tied, self.encoder)

        self.rnn = StackedRNN(rnn_type, emb_size, hidden_sizes, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        seq_len = output.size(0)
        batch_size = output.size(1)
        output = output.view(output.size(0) * output.size(1), output.size(2))

        decoded = self.decoder(output)
        return decoded.view(seq_len, batch_size, decoded.size(1)), hidden

    def init_hidden(self, bsz):
        return self.rnn.init_hidden(bsz)


class BidirectionalLanguageModel(nn.Module):
    """Container module with a (shared) encoder, two recurrent modules -- forward and backward -- and a decoder."""

    def __init__(self, word2idx, emb_size, hidden_sizes, dropout,
                 rnn_type="LSTM", pretrained_embs=None, fixed_embs=False, tied=None):
        super(BidirectionalLanguageModel, self).__init__()
        self.drop = nn.Dropout(dropout)

        self.encoder = Encoder(word2idx, emb_size, pretrained_embs, fixed_embs)
        self.decoder = Decoder(len(word2idx), hidden_sizes[-1], tied, self.encoder)

        self.forward_lstm = StackedRNN(rnn_type, emb_size, hidden_sizes, dropout)
        self.backward_lstm = StackedRNN(rnn_type, emb_size, hidden_sizes, dropout)

        self.rnn_type = rnn_type
        self.hidden_sizes = hidden_sizes
        self.nlayers = len(hidden_sizes)

    def forward(self, input, hidden):
        input_f, input_b = input
        emb_f = self.drop(self.encoder(input_f))
        emb_b = self.drop(self.encoder(input_b))

        hidden_f = hidden[0]
        hidden_b = hidden[1]

        output_f, hidden_f = self.forward_lstm(emb_f, hidden_f)
        output_b, hidden_b = self.backward_lstm(emb_b, hidden_b)

        output = output_f + utils.flip(output_b, dim=0)   # output is sum of forward and backward

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), (hidden_f, hidden_b)

    def init_hidden(self, bsz):
        return self.forward_lstm.init_hidden(bsz), self.backward_lstm.init_hidden(bsz)







