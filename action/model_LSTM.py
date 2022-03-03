import torch
from torch import nn

class ActNet(nn.Module):
    def __init__(self, nhid=100, nlayer=3):
        super(ActNet, self).__init__()
        # self.lstm = nn.LSTM(150, 100, 3, batch_first=True, dropout=0.5)
        self.lstm = nn.LSTM(150, nhid, nlayer, dropout=0.5)
        self.linear = nn.Sequential(nn.Linear(nhid, 60), nn.ELU())

    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        # print('inter input.shape=', inputs.shape)
        # inputs = inputs.transpose(0, 1)
        self.lstm.flatten_parameters()
        features, _ = self.lstm(inputs)  # B,T,F
        #features = features.permute(0,2,1)   # B,F,T
        #pool = nn.MaxPool1d(features.size()[2])
        #h = pool(features)
        #h = h.view(h.size(0),-1)
        # out = self.linear(features[:, -1, :])
        out = self.linear(features[-1, :, :])
        # out = self.linear(h)
        return out


class GenNet(nn.Module):
    def __init__(self, Num):
        super(GenNet, self).__init__()
        self.Enlstm = nn.LSTM(150, Num, 2, batch_first=True, dropout=0.5)
        self.Delstm = nn.LSTM(Num, 150, 2, batch_first=True, dropout=0.5)

    def forward(self, inputs):
        encoder, _ = self.Enlstm(inputs)
        decoder, _ = self.Delstm(encoder)
        return decoder
