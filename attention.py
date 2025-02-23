import torch
import torch.nn as nn
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear((hidden_size * 2), hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]

        seq_len = encoder_outputs.shape[1]

        # Lặp lại hidden để có kích thước [batch_size, seq_len, hidden_size]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_size]

        # Tính toán energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, seq_len, hidden_size]

        # Tính toán attention weights
        attention = self.v(energy).squeeze(2)  # [batch_size, seq_len]
        return torch.softmax(attention, dim=1)  # [batch_size, seq_len]