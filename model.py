# Copyright: Wentao Shi, 2021
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import softmax
from pad_idx import PADDING_IDX


class RNN(torch.nn.Module):
    def __init__(self, rnn_config):
        super(RNN, self).__init__()

        self.embedding_layer = nn.Embedding(
            num_embeddings=rnn_config['num_embeddings'],
            embedding_dim=rnn_config['embedding_dim'],
            padding_idx=PADDING_IDX
        )

        self.rnn = nn.LSTM(
            input_size=rnn_config['input_size'],
            hidden_size=rnn_config['hidden_size'],
            num_layers=rnn_config['num_layers'],
            batch_first=True,
            dropout=rnn_config['dropout']
        )

        # output does not include <sos> and <pad>, so
        # decrease the num_embeddings by 2
        self.linear = nn.Linear(
            rnn_config['hidden_size'], rnn_config['num_embeddings'] - 2)

    def forward(self, data, lengths):
        embeddings = self.embedding_layer(data)

        # pack the padded input
        # the lengths are decreased by 1 because we don't
        # use <eos> for input and we don't need <sos> for
        # output during traning.
        embeddings = pack_padded_sequence(
            input=embeddings, lengths=lengths, batch_first=True, enforce_sorted=False)

        # recurrent network, discard (h_n, c_n) in output.
        # Tearcher-forcing is used here, so we directly feed
        # the whole sequence to model.
        embeddings, _ = self.rnn(embeddings)

        # linear layer to generate input of softmax
        embeddings = self.linear(embeddings.data)

        # return the packed representation for backpropagation,
        # the targets will also be packed.
        return embeddings

    def sample(self, vocab):
        output = []

        # get integer of "start of sequence"
        start_int = vocab.vocab['<sos>']

        # create a tensor of shape [batch_size=1, seq_step=1]
        sos = torch.tensor(start_int).unsqueeze(
            dim=0).unsqueeze(dim=0)

        # sample first output
        x = self.embedding_layer(sos)
        x, (h, c) = self.rnn(x)
        x = self.linear(x)
        x = softmax(x, dim=-1)
        x = torch.multinomial(x.squeeze(), 1)
        output.append(x.item())

        # use first output to iteratively sample until <eos> occurs
        while output[-1] != vocab.vocab['<eos>']:
            x = x.unsqueeze(dim=0)
            x = self.embedding_layer(x)
            x, (h, c) = self.rnn(x, (h, c))
            x = self.linear(x)
            x = softmax(x, dim=-1)
            x = torch.multinomial(x.squeeze(), 1)
            output.append(x.item())

        # convert integers to tokens
        output = [vocab.int2tocken[x] for x in output]

        # popout <eos>
        output.pop()

        # convert to a single string
        output = vocab.combine_list(output)

        return output
