# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from data import loadview_data, loadsubject_data
# from model_LSTM_zoneout import ActNet, GenNet
from model_LSTM import ActNet, GenNet
import time
from torch.optim.lr_scheduler import StepLR
import os
import argparse
import numpy as np
import sys
sys.path.append("../../")
from TCN.action.model import *
# import model.MyRNNModel as MyRNNModel

parser = argparse.ArgumentParser(description='Sequence Modeling - Word-level Language Modeling')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (default: 0.45)')
parser.add_argument('--clip', type=float, default=0.0,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--data_mode', type=str, default='CS',
                    help='data mode, default is CS')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate (default: 0.001)')
parser.add_argument('--emsize', type=int, default=150,
                    help='size of word embeddings (default: 600)')
parser.add_argument('--nhid', type=int, default=100,  #700
                    help='number of hidden units per layer (default: 600)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--order', type=int, default=2,
                    help='order of LSTM')
parser.add_argument('--cell_type', type=str, default='ORG_MEMO',
                    help='memory cel type (default: ORG_MEMO')
parser.add_argument('--phase', type=str, default='TRAIN',
                    help='phase: TRAIN or TEST')
parser.add_argument('--sb', type=str, default='orglstm',
                    help='stride and block length')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--othermodel', action='store_true',
                    help='othermodel like zoneout etc. (default: True)')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--stride', type=int, default=1,
                    help='memory stride (default: 1)')
parser.add_argument('--bl', type=int, default=4,
                    help='block length (default: 4)')
parser.add_argument('--ups', type=int, default=0,
                    help='update stride (default: 0)')
parser.add_argument('--dm', type=float, default=0.2,
                    help='drop mask (default: 0.2)')
parser.add_argument('--my', nargs='+', type=int, default=0,
                    help='memory layer (default: [0])')
parser.add_argument('--ft', action='store_true',
                    help='fine-turn (default: False)')
args = parser.parse_args()
print(args)
dir_file = 'checkpoint/{}/myMEM_LSTM_{}_{}'.format(args.data_mode, args.nlayers, args.sb)
if not (os.path.exists(dir_file)):
    os.makedirs(dir_file)
# -------------------------------------------------------------------
# init data loder
print("*********")
batch_size = 256
data_path = './data/NTU_skeleton_main_lstm'
if args.data_mode == 'CV':
    print('load CV data ...')
    train_data, train_label, test_data, test_label  = loadview_data(data_path)
elif args.data_mode == 'CS':
    print('load CS data ...')
    train_data, train_label, test_data, test_label = loadsubject_data(data_path)
else:
    raise ValueError("data_mode must be CV or CS!")
dsets = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_label).long())
dsets_test = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_label).long())
dset_test_loaders = DataLoader(dataset=dsets_test, num_workers=4, batch_size=256, shuffle=False)
dsets_test_sizes = len(dsets_test)
torch.manual_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------
# init model
# mymodel = False
if not args.othermodel:
    model_ft = MyRNNModel(args.model, args.emsize, args.nhid, nlayers=args.nlayers, \
                    dropout=args.dropout, order=args.order, cell_type=args.cell_type, memory_layer=args.my, \
                    stride=args.stride, block_length=args.bl, dropmask=args.dm, update_stride=args.ups).to(device)
else:
    model_ft = ActNet().to(device)
print(model_ft)
if args.ft:
    print('load model ....')
    modelft_dir = 'checkpoint/{}/myMEM_LSTM_3_cv_s4b8_3layer_rebuttal_concatH_0616/'.format(args.data_mode)
    modelft_file1 = modelft_dir + 'MEM_LSTM_epoch_1249.pth'
    # modelft_file1 = modelft_dir + 'cv_s5b10_3layer_z2_wd001_mix01_256_0307_1349_8418.pth'
    model_ft = torch.load(modelft_file1)

criterion = nn.CrossEntropyLoss().to(device)
if args.cell_type != 'ORG_MEMO':
    print('change opti')
    optimizer = optim.Adam(model_ft.parameters(), lr=args.lr)  # 0.001
else:
    optimizer = optim.Adam(model_ft.parameters(), lr=args.lr, weight_decay=args.wd)  # 0.001
# optimizer = optim.Adam(model_ft.parameters(), lr=args.lr)  # 0.001
scheduler = StepLR(optimizer, step_size=600, gamma=0.1)

######################################################################
allloss = []

def test_model(model_ft, criterion):
    model_ft.eval()
    running_loss = 0.0
    running_corrects = 0
    running_errors = 0
    cont = 0
    Deinputs_array_total = []
    De_class_array_total = []
    out_class_array_total = []
    # Iterate over data.
    for data in dset_test_loaders:
        # get the inputs
        inputs, labels = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs, labels = inputs.to(device), labels.to(device)
        # forward
        out_class = model_ft(inputs)
        out_class_array = out_class.data.cpu().numpy()
        if out_class_array_total == []:
            out_class_array_total = np.vstack((out_class_array))
        else:
            out_class_array_total = np.vstack((out_class_array_total, out_class_array))

        _, preds = torch.max(out_class.data, 1)

        loss = criterion(out_class, labels)
        if cont == 0:
            outPre = out_class.data.cpu()
        else:
            outPre = torch.cat((outPre, out_class.data.cpu()), 0)
        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
        running_errors += torch.sum(preds != labels.data)
        # print('Num:', cont)
        cont += 1
    # print(Deinputs_array_total.shape)
    # scio.savemat('ende_100.mat', {'pred': Deinputs_array_total})

    print('Testing: Loss: {:.4f} Acc: {:.4f} Err: {:.4f}'.format(running_loss / dsets_test_sizes,
                                                        running_corrects.cpu().numpy() / (1.0 * dsets_test_sizes),
                                                        running_errors.cpu().numpy() / (1.0 * dsets_test_sizes)))
    model_ft.train()
    return outPre


def train_model(model_ft, criterion, optimizer, scheduler, num_epochs=60):
    since = time.time()
    model_ft.train()
    dset_sizes = len(dsets)
    print('Data Size', dset_sizes)
    for epoch in range(num_epochs):

        dset_loaders = DataLoader(dataset=dsets, num_workers=4, batch_size=batch_size, shuffle=True)
        # Each epoch has a training and validation phase
        # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        count = 0
        scheduler.step()
        # Iterate over data.
        for data in dset_loaders:
            # get the inputs
            inputs, labels = data
            labels, inputs = labels.to(device), inputs.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            out_class = model_ft(inputs)
            loss_out_class = criterion(out_class, labels)
            loss = loss_out_class

            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model_ft.parameters(), args.clip)
            optimizer.step()

            _, preds = torch.max(out_class.data, 1)
            # backward + optimize only if in training phase
            count += 1
            if count % 10 == 0 or inputs.size()[0] < batch_size:
                # print('Epoch:{}:loss_out_class:{:.3f}'.format(epoch, loss_out_class.item()))
                allloss.append(loss.item())

            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.cpu().numpy() / (dset_sizes * 1.0)
        # print('Training: Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        print('Epoch {}/{} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch, num_epochs - 1, epoch_loss, epoch_acc))
        if (epoch + 1) % 50 == 0:
            model_out_path = dir_file + "/MEM_LSTM_epoch_{}.pth".format(epoch)
            time.sleep(10)
            torch.save(model_ft, model_out_path)
            with torch.no_grad():
                _ = test_model(model_ft, criterion)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return allloss


if __name__ == "__main__":
    if args.phase == 'TRAIN':
        allloss = train_model(model_ft, criterion, optimizer, scheduler, num_epochs=10000)
    else:
        modelft_dir = 'checkpoint/{}/myMEM_LSTM_1/'.format(args.data_mode)
        modelft_file1 = modelft_dir + 'MEM_LSTM_epoch_49.pth'
        model_ft = torch.load(modelft_file1).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        outPre = test_model(model_ft, criterion) 
