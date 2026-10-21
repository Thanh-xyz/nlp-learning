import torch
import torch.nn as nn
import math

class SimpleRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(SimpleRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.has_bias = bias

        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x_t, h_prev=None):
        if h_prev is None:
            batch_size = x_t.size(0) if x_t.dim() > 1 else 1
            h_prev = torch.zeros(batch_size, self.hidden_size, dtype=x_t.dtype, device=x_t.device)
            if x_t.dim() == 1:
                h_prev = h_prev.squeeze(0)

        linear_input = torch.matmul(x_t, self.weight_ih.t())
        linear_hidden = torch.matmul(h_prev, self.weight_hh.t())
        pre_activation = linear_input + linear_hidden

        if self.has_bias:
            pre_activation = pre_activation + self.bias_ih + self.bias_hh

        h_t = torch.tanh(pre_activation)
        return h_t
