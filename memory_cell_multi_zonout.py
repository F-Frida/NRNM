import torch
import torch.nn as nn
import math

from torch.nn import init
from TCN.action.locked_dropout import LockedDropout

memory_version = 6
print('memory_version = ',memory_version)
if memory_version == 1:
    from mem_v1 import MemoryCell
if memory_version == 2:
    from mem_v2 import MemoryCell  # only hidden
if memory_version == 3:
    from mem_v3 import MemoryCell  # use memory and pre_memory
if memory_version == 4:
    from mem_v4 import MemoryCell  # use memory and pre_memory
if memory_version == 5:
    from mem_v5 import MemoryCell  # make a hidden projector
if memory_version == 6:
    from mem_v6 import MemoryCell  # make a hidden projector
if memory_version == 7:
    from mem_v7 import MemoryCell  # make a hidden projector
if memory_version == 8:
    from mem_v8 import MemoryCell  # make a hidden projector
if memory_version == 9:
    from mem_v9 import MemoryCell  # make a hidden projector
if memory_version == 10:
    from mem_v10 import MemoryCell  # make a hidden projector
if memory_version == 11:
    from mem_v11 import MemoryCell  # make a hidden projector
if memory_version == 12:
    from mem_v12 import MemoryCell  # make a hidden projector
if memory_version == 15:
    from mem_v15 import MemoryCell  # make a hidden projector
input_list = None

class LSTMCell(nn.Module):
    """A basic LSTM cell."""
    def __init__(self, input_size, hidden_size, order=1, stride=1, dropmask=0.2, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.order = order
        self.stride = stride
        self.input_list = None
        self.dropout_mo = 0.5
        self.dropout_me = 0.5
        self.dropmask = dropmask
        self.meoutput_lockdrop = nn.Dropout(self.dropout_mo)
        self.memory_lockdrop = LockedDropout()
        self.head_size = self.hidden_size // 4
        assert self.head_size * 4 == self.hidden_size

        self.MC = MemoryCell(mem_slots=self.order,  # maybe should be equal to high order size , 1 for memory_version=5, others, sell.order
                             head_size=self.head_size,
                             input_size=input_size,
                             num_heads=4,
                             num_blocks=1,
                             forget_bias=1.,
                             input_bias=0.) # .cuda()
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size,  4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        global memory_version
        if memory_version in [5, 9, 11]:
            print('memory_version 5, 9, 11 init weight')
            self.weight_memh = nn.Parameter(
                torch.FloatTensor(hidden_size, hidden_size))
            self.weight_memt = nn.Parameter(
                torch.FloatTensor(hidden_size, hidden_size))
            self.weight_memi = nn.Parameter(
                torch.FloatTensor(input_size, hidden_size))   # hidden_size -> input_size
        else:
            self.weight_memh = nn.Parameter(
                torch.FloatTensor(self.order * hidden_size, self.order * hidden_size))
            self.weight_memt = nn.Parameter(
                torch.FloatTensor(self.order * hidden_size, hidden_size))
            self.weight_memi = nn.Parameter(
                torch.FloatTensor(input_size, self.order * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
            if memory_version in [9, 11]:
                self.mem_bias = nn.Parameter(torch.FloatTensor(hidden_size))
            else:
                self.mem_bias = nn.Parameter(torch.FloatTensor(self.order * hidden_size))
        else:
            self.register_parameter('bias', None)
        self._dropout_mask = None
        self.reset_parameters()

    def old_reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        init.orthogonal_(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        if self.use_bias:
            init.constant(self.bias.data, val=0)

    def set_dropout_mask(self, batch_size, device):
        if self.training:
            self._dropout_mask = torch.bernoulli(
                torch.Tensor(self.hidden_size).fill_(1 - self.dropmask)).to(device)
        else:
            self._dropout_mask = 1 - self.dropmask

    def reset_parameters(self):   # official_reset_parameters
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hx, me_output, memory, time):
        global input_list
        temp = input_list
        if input_list is None:
            input_list = input_
        else:
            if time % self.stride == 0:
                if input_list.size(1)/input_.size(1) == self.order:
                    self.input_cat = input_list
                    input_list = input_
                else:
                    input_list = torch.cat((input_list, input_), 1)
        hx_pre = hx[0]
        if time % self.stride == 0:   # for each stride, we should update hx_output
            hx_output = hx[1:-1]
        else:
            hx_output = hx[1:]

        batch_size = hx_pre[0].size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))     # [3,12]
        mem_bias_batch = (self.mem_bias.unsqueeze(0)
                          .expand(batch_size, *self.mem_bias.size()))

        flag = time % (self.order * self.stride)
        if flag == 0:
            if time > 0:
                # print(self.input_cat.size(1)/input_.size(1))
                assert self.input_cat.size(1) / input_.size(1) == self.order
                if time % self.stride == 0:
                    hiddens = torch.stack([hi[0] for hi in hx_output] + [hx[-1][0]], dim=1)
                else:
                    hiddens = torch.stack([hi[0] for hi in hx_output], dim=1)
                me_output, next_memory = self.MC(self.input_cat, memory, hiddens, treat_input_as_matrix=False)
                # me_output = self.meoutput_lockdrop(me_output)
                next_memory = self.memory_lockdrop(next_memory, self.dropout_me)
                memory = next_memory
        if time % self.stride == 1:
            # when time % stride equal to 1, hx[0] is equal to hx[1]
            wh_b = torch.addmm(bias_batch, hx_output[0][0], self.weight_hh)  # [3, 12]    # bias_batch + h_0 * weight_hh
            wi = torch.mm(input_, self.weight_ih)  # [3, 12]               # input
            f, i, o, g = torch.split(wh_b + wi,
                                     self.hidden_size, dim=1)

            memory_gate = torch.mm(input_, self.weight_memi) + torch.mm(me_output, self.weight_memh) + mem_bias_batch
            if memory_version in [5, 9, 11]:
                c_next = torch.sigmoid(f) * hx_output[0][1] + torch.sigmoid(i) * torch.tanh(g) \
                         + torch.sigmoid(memory_gate) * me_output
            else:
                c_next = torch.sigmoid(f) * hx_output[0][1] + torch.sigmoid(i) * torch.tanh(g) \
                         + torch.mm(torch.sigmoid(memory_gate) * me_output, self.weight_memt)
            # c_next = torch.sigmoid(f)*hx_output[0][1] + torch.sigmoid(i)*torch.tanh(g) + torch.mm(me_output, self.weight_mem)
            h_next = torch.sigmoid(o) * torch.tanh(c_next)
            # zonout drop
            h_next = h_next * self._dropout_mask + hx_output[0][0] * (1 - self._dropout_mask)
            c_next = c_next * self._dropout_mask + hx_output[0][1] * (1 - self._dropout_mask)
        else:
            wh_b = torch.addmm(bias_batch, hx_pre[0], self.weight_hh)  # [3, 12]    # bias_batch + h_0 * weight_hh
            wi = torch.mm(input_, self.weight_ih)     # [3, 12]               # input
            f, i, o, g = torch.split(wh_b + wi,
                                     self.hidden_size, dim=1)

            memory_gate = torch.mm(input_, self.weight_memi) + torch.mm(me_output, self.weight_memh) + mem_bias_batch
            if memory_version in [5, 9, 11]:
                c_next = torch.sigmoid(f)*hx_pre[1] + torch.sigmoid(i)*torch.tanh(g) \
                         + torch.sigmoid(memory_gate) * me_output
            else:
                c_next = torch.sigmoid(f)*hx_pre[1] + torch.sigmoid(i)*torch.tanh(g) \
                         + torch.mm(torch.sigmoid(memory_gate) * me_output, self.weight_memt)
            # c_next = torch.sigmoid(f)*hx_output[0][1] + torch.sigmoid(i)*torch.tanh(g) + torch.mm(me_output, self.weight_mem)
            h_next = torch.sigmoid(o) * torch.tanh(c_next)
            h_next = h_next * self._dropout_mask + hx_pre[0] * (1 - self._dropout_mask)
            c_next = c_next * self._dropout_mask + hx_pre[1] * (1 - self._dropout_mask)
        return h_next, c_next, hx_output, me_output, memory

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""
    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, batch_size=256, dropouth=0.25,  **kwargs):
        super(LSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.me_output_init = False
        self.memory = None
        self.me_output = None
        self.hx0 = None
        self.batch_size = batch_size
        self.version = 2
        self.lockdrop = LockedDropout()
        self.dropouth = dropouth
        self.block_length = 10
        self.stride = 5
        print('version =', self.version)
        print('stride =', self.stride)
        print('block_length =', self.block_length)

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              order=self.block_length//self.stride,
                              stride=self.stride,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()
        print("init memory_cell_multi_zonout!")

    def reset_hidden(self):
        self.hx0 = None
        self.me_output_init = False
        self.me_output = None
        self.memory = None
        global input_list
        input_list = None

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def _forward_rnn(self, cell, input_, me_output, memory, length, hx):
        max_time, batch_size, _ = input_.size()
        output = []
        cell.set_dropout_mask(batch_size, input_.get_device())
        for time in range(max_time):
            if (not self.hx0) or (time == 0):
                hx0 = input_.data.new(batch_size, self.hidden_size).zero_()
                _hx = []
                _hx.append(hx)
                _hx.append((hx0, hx0))
                for i in range(self.block_length//self.stride - 1):   # wo do not need block_length memory
                    _hx.append((hx0, hx0))
                self.hx0 = True
            else:
                _hx = hx
            input = input_[time]
            h_next, c_next, hx_output, me_output, memory = cell(input_=input,
                                                                hx=_hx,
                                                                me_output=me_output,
                                                                memory=memory,
                                                                time=time)
            mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            h_next = h_next*mask + _hx[0][0]*(1 - mask)
            c_next = c_next*mask + _hx[0][1]*(1 - mask)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = []
            if time % self.stride == 0:   # the top must be hx_next!!!
                hx.append(hx_next)
                hx.append(hx_next)
                hx.extend(hx_output)
            else:
                hx.append(hx_next)
                hx.extend(hx_output)
        output = torch.stack(output, 0)
        return output, hx, me_output, memory

    def forward(self, input_, hx, length=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = torch.LongTensor([max_time] * batch_size)
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
        if hx is None:
            hx = input_.data.new(batch_size, self.hidden_size).zero_()
            hx = (hx, hx)
        if not self.me_output_init:
            self.me_output_init = True
            cell = self.get_cell(0)
            cell_order = cell.order
            mem_slots = cell.MC.mem_slots
            mem_size = cell.MC.head_size * cell.MC.num_heads
            # print('cell_order, mem_slots, mem_size = ', cell_order, mem_slots, mem_size)
            # maybe should be [layer, batch, hidden*cell_order]
            if self.me_output is None or self.memory is None:
                if self.version == 1:
                    self.me_output = input_.data.new(batch_size, self.hidden_size * cell_order).zero_()
                    self.memory = input_.data.new(batch_size, mem_slots, mem_size).zero_()
                if self.version == 2:
                    global memory_version
                    if memory_version in [5, 9, 11]:
                        _me_output = input_.data.new(batch_size, self.hidden_size).zero_()
                    else:
                        _me_output = input_.data.new(batch_size, self.hidden_size * cell_order).zero_()
                    _memory = input_.data.new(batch_size, mem_slots, mem_size).zero_()
                    self.me_output = [_me_output]
                    self.memory = [_memory]
                    for i in range(1, self.num_layers):
                        self.me_output.append(_me_output)
                        self.memory.append(_memory)
        h_n = []
        c_n = []
        if self.version == 1:
            me_output = self.me_output
            memory = self.memory
        layer_output = None
        for layer in range(self.num_layers):
            if self.version == 2:
                me_output = self.me_output[layer]
                memory = self.memory[layer]
                global input_list
                input_list = None  # for each layer, clear inout_list
            cell = self.get_cell(layer)
            hidden_shape = hx[0].shape
            if len(hidden_shape) > 2:   # for multi-layer
                _hx = (hx[0][layer], hx[1][layer])
            else:
                _hx = hx
            layer_output, hx_out, me_output, memory = self._forward_rnn(
                cell=cell, input_=input_, me_output=me_output, memory=memory, length=length, hx=_hx)
            # input_ = self.dropout_layer(layer_output)   # add 2019/2/21
            input_ = self.lockdrop(layer_output, self.dropouth)
            input_ = layer_output
            h_n.append(hx_out[0][0])
            c_n.append(hx_out[0][1])
            if self.version == 2:
                self.me_output[layer] = me_output.detach()  # for each batch, detach memory and me_output from graph
                self.memory[layer] = memory.detach()

        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        return output, (h_n, c_n)

if __name__ == '__main__':
    layer = 2
    torch.manual_seed(1000)
    input = torch.randn(40, 20, 512)
    lstm = LSTM(LSTMCell, 512, 512, layer)
    h0 = torch.zeros(layer, 20, 512)
    c0 = torch.zeros(layer, 20, 512)

    out, (hn, cn) = lstm(input, hx=(h0, c0))
    print(out.shape, hn[0].shape)
