import torch
import torch.nn as nn
from attention import Attention
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        seq_len = encoder_outputs.shape[1]
        hidden_expanded = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attention(torch.cat((hidden_expanded, encoder_outputs), dim=2)))
        attention_weights = torch.softmax(self.v(energy).squeeze(2), dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell



