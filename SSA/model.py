import torch
import torch.nn.functional as F
from torch import nn


class StructuredSelfAttention(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, n_layers, n_hop, n_class, bidirectional=True):
        super(StructuredSelfAttention, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_hop = n_hop
        self.n_class = n_class
        self.bidirectional = bidirectional


        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)

        self.self_attn = _SSA(input_size=hidden_size*2, hidden_size=hidden_size, n_hop=n_hop)
        self.identity_matrix = torch.eye(n_hop, hidden_size*2, dtype=torch.float, requires_grad=False)

        self.MLP = self._create_linear(hidden_size, 3, 100, n_class)

    def forward(self, input_seq):
        rnn_output, _ = self.lstm(input_seq)
        attn_matrix = self.self_attn(rnn_output)
        attn_matrix = attn_matrix.transpose(1, 2)

        output = torch.bmm(attn_matrix, rnn_output)
        penalization_term = output - self.identity_matrix

        return output, penalization_term

    @staticmethod
    def _create_linear(input_size, nlayers, hidden_size, output_size):
        if nlayers == 1:
            return nn.Linear(input_size, output_size)
        elif nlayers == 2:
            return nn.Sequential(nn.Linear(input_size, hidden_size), nn.Linear(hidden_size, output_size))
        else:
            mlp = [nn.Linear(input_size, hidden_size)]
            mlp += [nn.Linear(hidden_size, hidden_size) for i in range(nlayers-2)]
            mlp.append(nn.Linear(hidden_size, output_size))
            return nn.Sequential(*mlp)


class _SSA(nn.Module):
    def __init__(self, input_size, hidden_size, n_hop):
        super(_SSA, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size, bias=False)
        self.output_layer = nn.Linear(hidden_size, n_hop, bias=False)

    def forward(self, input_seq):
        """
        :param input_seq: batch x sequence_len x hidden_size
        :return:
        """
        output = torch.tanh(self.input_layer(input_seq))
        output = self.output_layer(output)
        output = F.softmax(output, dim=1)
        return output


def main():
    batch = 2
    seq_len = 10
    n_embed = 20
    hidden_size = 30

    model = StructuredSelfAttention(batch, n_embed, hidden_size, n_layers=3, n_hop=40, n_class=2)

    input_seq = torch.randn(batch, seq_len, n_embed)
    output = model(input_seq)


if __name__ == '__main__':
    main()
