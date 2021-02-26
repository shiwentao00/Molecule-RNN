import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class RNN(torch.nn.Module):
    def __init__(self, rnn_config):
        super(RNN, self).__init__()

        self.embedding_layer = nn.Embedding(
            num_embeddings=rnn_config['num_embeddings'],
            embedding_dim=rnn_config['embedding_dim'],
            padding_idx=0
        )

        self.rnn = nn.LSTM(
            input_size=rnn_config['input_size'],
            hidden_size=rnn_config['hidden_size'],
            num_layers=rnn_config['num_layers'],
            batch_first=True,
            dropout=rnn_config['dropout']
        )

        self.linear = nn.Linear(
            rnn_config['hidden_size'], rnn_config['num_embeddings'])

    def forward(self, data, lengths):
        embeddings = self.embedding_layer(data)
        # print(embeddings.size())
        # print(graph_embedding.size())

        # pack the padded input
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

    def sample(self):
        # TO-DO
        pass

    def beam_search(self):
        # TO-DO
        pass
