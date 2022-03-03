import torch
import torch.nn as nn
import math


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        # return h.detach().requires_grad_()
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class HRORNNCell(nn.Module):
    """A basic High Order RNN cell."""
    def __init__(self, input_size, hidden_size, order=1, use_bias=True):
        super(HRORNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.order = order
        print('order =', self.order)
        self.weight_ih = nn.Linear(input_size, hidden_size)
        # self.weight_hh = nn.Parameter(torch.FloatTensor(self.order * hidden_size, hidden_size))
        self.weight_hh_list = nn.ModuleList(nn.Linear(hidden_size, hidden_size)
                                       for i in range(self.order))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):  # official_reset_parameters
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hx):
        # print(len(hx[0]))
        assert len(hx) == self.order
        wh_ = 0
        for i in range(self.order):
            wh_ = wh_ + self.weight_hh_list[i](hx[i])
        # wh_b = torch.add(bias_batch, wh_)
        wi = self.weight_ih(input_)
        h_next = torch.tanh(torch.add(wh_, wi))

        return h_next, hx[:-1]

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class HRORNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, cell_class=HRORNNCell,
                 use_bias=True, batch_first=False, dropout=0, order=1, **kwargs):
        super(HRORNN, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.hx0 = False
        self.hx_pre = None
        self.hx_pre_pre = None
        self.order = order

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              order=self.order,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_hidden(self):
        self.hx0 = False
        self.f0 = None

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def _forward_rnn(self, cell, input_, length, hx):
        max_time, batch_size, _ = input_.size()
        # print('shape =', input_.size())
        output = []
        for time in range(max_time):
            if (not self.hx0) or time == 0:
                hx0 = input_.data.new(batch_size, self.hidden_size).zero_()
                _hx = []
                for i in range(self.order):
                    _hx.append(hx0)
                self.hx0 = True
            else:
                _hx = hx
            input = input_[time]
            h_next, h_pre = cell(input_=input, hx=_hx)
            mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            h_next = h_next * mask + _hx[0] * (1 - mask)
            output.append(h_next)

            hx = []
            hx.append(h_next)
            hx.extend(h_pre)
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, hx, length=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = torch.LongTensor([max_time] * batch_size)
            if input_.is_cuda:
                device = input_.get_device()
                length = length.to(device)
        if hx is None:
            hx = input_.data.new(self.num_layers, batch_size, self.hidden_size).zero_()
        h_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            if layer == 0:  # for init
                hidden_shape = hx.shape
                if len(hidden_shape) > 2:  # for multi-layer, only for layer_0
                    _hx = hx[layer]
                else:
                    _hx = hx
            else:
                _hx = hx
            layer_output, hx = self._forward_rnn(
                cell=cell, input_=input_, length=length, hx=_hx)
            input_ = self.dropout_layer(layer_output)
            h_n.append(hx[0])

        output = layer_output
        h_n = torch.stack(h_n, 0)
        # print('h_n.size()=', h_n.size())
        self.hx0 = False  # a temporary solution
        return output, h_n


if __name__ == '__main__':
    layer = 2
    torch.manual_seed(1000)
    input = torch.randn(40, 16, 700)
    lstm = HRORNN(700, 700, layer)
    h0 = torch.randn(layer, 16, 700)

    out, hn = lstm(input, hx=h0)
    print(out.shape, hn.shape)
