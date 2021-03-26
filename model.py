# Copyright: Wentao Shi, 2021
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import softmax


class RNN(torch.nn.Module):
    def __init__(self, rnn_config):
        super(RNN, self).__init__()

        self.embedding_layer = nn.Embedding(
            num_embeddings=rnn_config['num_embeddings'],
            embedding_dim=rnn_config['embedding_dim'],
            padding_idx=rnn_config['num_embeddings'] - 2
        )

        if rnn_config['rnn_type'] == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=rnn_config['input_size'],
                hidden_size=rnn_config['hidden_size'],
                num_layers=rnn_config['num_layers'],
                batch_first=True,
                dropout=rnn_config['dropout']
            )
        elif rnn_config['rnn_type'] == 'GRU':
            self.rnn = nn.GRU(
                input_size=rnn_config['input_size'],
                hidden_size=rnn_config['hidden_size'],
                num_layers=rnn_config['num_layers'],
                batch_first=True,
                dropout=rnn_config['dropout']
            )
        else:
            raise ValueError(
                "rnn_type should be either 'LSTM' or 'GRU'."
            )

        # output does not include <sos>, so
        # decrease the num_embeddings by 1
        self.linear = nn.Linear(
            rnn_config['hidden_size'], rnn_config['num_embeddings'] - 1)

    def forward(self, x):
        # remove last tokens which are <eos> or <pad>
        x = x[:, 0:-1]

        x = self.embedding_layer(x)

        # recurrent network, discard (h_n, c_n) in output.
        # Tearcher-forcing is used here, so we directly feed
        # the whole sequence to model.
        x, _ = self.rnn(x)

        # linear layer to generate input of softmax
        x = self.linear(x)

        return x

    def sample(self, batch_size, vocab, device, max_length=140):
        """Use this function if device is GPU"""
        # get integer of "start of sequence"
        start_int = vocab.vocab['<sos>']

        # create a tensor of shape [batch_size, seq_step=1]
        sos = torch.ones(
            [batch_size, 1], 
            dtype=torch.long, 
            device=device
        )
        sos = sos * start_int

        # sample first output
        output = []
        x = self.embedding_layer(sos)
        x, hidden = self.rnn(x)
        x = self.linear(x)
        x = softmax(x, dim=-1)
        x = torch.multinomial(x.squeeze(), 1)
        output.append(x)

        # a tensor to indicate if the <eos> token is found
        # for all data in the mini-batch
        finish = torch.zeros(batch_size, dtype=torch.bool).to(device)

        # sample until every sequence in the mini-batch
        # has <eos> token
        for _ in range(max_length):
            # forward
            x = self.embedding_layer(x)
            #print(x.size())
            x, hidden = self.rnn(x, hidden)
            x = self.linear(x)
            x = softmax(x, dim=-1)
            x = torch.multinomial(x.squeeze(), 1)
            output.append(x)

            # terminate if <eos> is found for every data
            eos_sampled = (x == vocab.vocab['<eos>']).data
            finish = torch.logical_or(finish, eos_sampled.squeeze())
            if torch.all(finish): break

        return torch.cat(output, -1)

    def sample_cpu(self, vocab):
        """Use this function if device is CPU"""
        output = []

        # get integer of "start of sequence"
        start_int = vocab.vocab['<sos>']

        # create a tensor of shape [batch_size=1, seq_step=1]
        sos = torch.tensor(
            start_int, 
            dtype=torch.long
        ).unsqueeze(dim=0
        ).unsqueeze(dim=0)

        # sample first output
        x = self.embedding_layer(sos)
        x, hidden = self.rnn(x)
        x = self.linear(x)
        x = softmax(x, dim=-1)
        x = torch.multinomial(x.squeeze(), 1)
        output.append(x.item())

        # use first output to iteratively sample until <eos> occurs
        while output[-1] != vocab.vocab['<eos>']:
            x = x.unsqueeze(dim=0)
            x = self.embedding_layer(x)
            x, hidden = self.rnn(x, hidden)
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
