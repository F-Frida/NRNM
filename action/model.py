import torch
from torch import nn
import sys
sys.path.append("../../")
import TCN.action.cell as cell
from TCN.action.locked_dropout import LockedDropout
import TCN.action.ho_rnn as ho_rnn
import TCN.action.EleAtt_LSTM as EleAtt_LSTM
import TCN.action.EleAtt_GRU as EleAtt_GRU
memory_cell_type = 9
print('memory_cell_type =', memory_cell_type)
if memory_cell_type == 1:
    import TCN.action.memory_cell as memory_cell
if memory_cell_type == 2:
    import TCN.action.memory_cell_v2 as memory_cell
if memory_cell_type == 3:
    import TCN.action.memory_cell_multi as memory_cell
if memory_cell_type == 4:
    import TCN.action.memory_cell_multi_v2 as memory_cell
if memory_cell_type == 5:
    import TCN.action.memory_cell_multi_v3 as memory_cell
if memory_cell_type == 6:
    import TCN.action.memory_cell_multi_v4 as memory_cell
if memory_cell_type == 7:
    import TCN.action.memory_cell_multi_zonout as memory_cell
if memory_cell_type == 8:
    import TCN.action.memory_cell_mixm as memory_cell
if memory_cell_type == 9:
    import TCN.action.memory_cell_mixmzonout as memory_cell
if memory_cell_type == 10:
    import TCN.action.memory_cell_mixmzonout_memorymask as memory_cell
if memory_cell_type == 11:   # for draw fig.
    import TCN.action.memory_cell_view as memory_cell

class MyRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, ninp, nhid, nlayers, nClass=60, dropout=0.5, order=2, 
                 cell_type='HO', memory_layer=[0,1], stride=1, block_length=4, dropmask=0.2, update_stride=0):
        super(MyRNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.lockdrop = LockedDropout()

        self.cell_type = cell_type
        print(self.cell_type)
        if rnn_type in ['LSTM']:
            # self.rnn = getattr(cell, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            if self.cell_type == 'HO':
                print('high order rnn')
                assert order > 1
                # self.rnn = holstm_multi.HROLSTM(ninp, nhid, order, nlayers, dropout=dropout)
                # self.rnn = ho_rnn.HRORNN(ninp, nhid, order, nlayers, dropout=dropout)
                self.rnn = ho_rnn.HRORNN(ninp, nhid, order=order, num_layers=nlayers, dropout=dropout)
            elif self.cell_type == 'CELL':
                print('high order and cell')
                self.rnn = holstm_cell.HROLSTM(holstm_cell.HROLSTMCell, ninp, nhid, nlayers, dropout=dropout)
            elif self.cell_type == 'BLOCK':
                print('block')
                self.rnn = holstm_block.HROLSTM(holstm_block.HROLSTMCell, ninp, nhid, nlayers, dropout=dropout)
            elif self.cell_type == 'M_BLOCK':
                print('m_block')
                self.rnn = holstm_memoryblock.HROLSTM(holstm_memoryblock.HROLSTMCell, ninp, nhid, nlayers, dropout=dropout)
            elif self.cell_type == 'HO_ORG':
                print('high order org')
                self.rnn = holstm.HROLSTM(holstm.HROLSTMCell, ninp, nhid, nlayers, dropout=dropout)
            elif self.cell_type == 'HO_SKIP':
                print('high order skip')
                self.rnn = holstm_skip.HROLSTM(holstm_skip.HROLSTMCell, ninp, nhid, nlayers, dropout=dropout)
            elif self.cell_type == '3HO':
                print('3HO')
                self.rnn = holstm_3.HROLSTM(holstm_3.HROLSTMCell, ninp, nhid, nlayers, dropout=dropout)
                print(self.rnn)
            elif self.cell_type == 'ORG_LSTM':
                print('org lstm')
                self.rnn = cell.LSTM(cell.LSTMCell, ninp, nhid, nlayers, dropout=dropout)
            elif self.cell_type == 'ELEATT_LSTM':
                print('ele att lstm')
                self.rnn = EleAtt_LSTM.LSTM(EleAtt_LSTM.EleAttLSTMCell, ninp, nhid, nlayers, dropout=dropout)
            elif self.cell_type == 'ORG_MEMO':
                print('org memory')
                if memory_cell_type == 9 or memory_cell_type==11:
                    self.rnn = memory_cell.LSTM(memory_cell.LSTMCell, ninp, nhid, nlayers, 
                                     dropout=dropout, memory_layer=memory_layer, stride=stride, 
                                     block_length=block_length, dropmask=dropmask, update_stride=update_stride)
                else:
                    self.rnn = memory_cell.LSTM(memory_cell.LSTMCell, ninp, nhid, nlayers, dropout=dropout)
            else:
                print('error!')
        elif rnn_type in ['GRU']:
            if self.cell_type == 'ORG_GRU':
                self.rnn = nn.GRU(ninp, nhid, nlayers, dropout=dropout)
            elif self.cell_type == 'ELEATT_GRU':
                print('ele att gru')
                self.rnn = EleAtt_GRU.GRU(EleAtt_GRU.EleAttGRUCell, ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        # self.decoder = nn.Linear(nhid, nClass)
        self.decoder = nn.Sequential(nn.Linear(nhid, nClass), nn.ELU())

        # self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        #stdv = 1.0 / math.sqrt(self.hidden_size)
        #for weight in self.parameters():
        #    init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hidden=None):
        # print('test')
        input.transpose_(0, 1)   # B,T,L -> T,B,L
        if hidden is None:
            hidden = self.init_hidden(input.size()[1])
        output, hidden = self.rnn(input, hidden)
        # output = self.lockdrop(output, self.dropouto)
        # output = self.drop(output)  # [40, 16, 700]
        decoded = self.decoder(output[-1,:,:])  # [640, 10000]
        return decoded

    def init_hidden(self, bsz):
        if self.cell_type in ['HO','HO_ORG', '3HO', 'BLOCK', 'M_BLOCK', 'HO_SKIP', 'ORG_MEMO']:
           self.rnn.reset_hidden() 
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM' and self.cell_type not in ['HO']:
            temp = weight.new_zeros(self.nlayers, bsz, self.nhid)
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
