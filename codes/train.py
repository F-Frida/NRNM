# -*- coding: utf-8 -*-

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from data import loadview_data, loadsubject_data
# from model_LSTM_zoneout import ActNet, GenNet
from model_LSTM import ActNet, GenNet
import time
from torch.optim.lr_scheduler import StepLR
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Compress Num")
parser.add_argument('--Num', type=int, default=75)
args = parser.parse_args()
Num = args.Num
dir_file = 'model/subject/3LSTM_MAX_{}'.format(Num)
if not (os.path.exists(dir_file)):
    os.makedirs(dir_file)
######################################
print("*********")
batch_size = 256
data_path = './data/NTU_skeleton_main_lstm'
data_mode = 'CS'
print('data_mode =', data_mode)
if data_mode == 'CV':
    train_data, train_label, test_data, test_label  = loadview_data(data_path)
elif data_mode == 'CS':
    train_data, train_label, test_data, test_label = loadsubject_data(data_path)
else:
    raise ValueError("data_mode must be CV or CS")
dsets = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_label).long())

# test dataset
dsets_test = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_label).long())
dset_test_loaders = DataLoader(dataset=dsets_test, num_workers=4, batch_size=256, shuffle=False)
dsets_test_sizes = len(dsets_test)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

        # three test methods
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
        #print('Num:', cont)
        cont += 1
    # print(Deinputs_array_total.shape)
    # scio.savemat('ende_100.mat', {'pred': Deinputs_array_total})

    print('Testing: Loss: {:.4f} Acc: {:.4f} Err: {:.4f}'.format(running_loss / dsets_test_sizes,
                                                        running_corrects.cpu().numpy() / (1.0 * dsets_test_sizes),
                                                        running_errors.cpu().numpy() / (1.0 * dsets_test_sizes)))
    model_ft.train()
    return outPre

def train_model(model_ft, criterion, MSEdis, optimizer, scheduler, num_epochs=60):
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
            # train
            out_class = model_ft(inputs)

            loss_out_class = criterion(out_class, labels)
            loss = loss_out_class
            loss.backward()
            optimizer.step()
            _, preds = torch.max(out_class.data, 1)
            # backward + optimize only if in training phase
            count += 1
            if count % 50 == 0 or inputs.size()[0] < batch_size:
                # print('Epoch:{}:loss_out_class:{:.3f}'.format(epoch, loss_out_class.item()))
                allloss.append(loss.item())

                # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.cpu().numpy() / (dset_sizes * 1.0)

        print('Epoch {}/{} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch, num_epochs - 1, epoch_loss, epoch_acc))

        if (epoch + 1) % 50 == 0:
            model_out_path = dir_file + "/ClassLSTM_epoch_{}.pth".format(epoch)
            torch.save(model_ft, model_out_path)
            _ = test_model(model_ft, criterion)

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return allloss


######################################################################
model_ft = ActNet().to(device)

criterion = nn.CrossEntropyLoss().to(device)
MSEdis = nn.MSELoss().to(device)
optimizer = optim.Adam(model_ft.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=400, gamma=0.1)
######################################################################
# Train
allloss = train_model(model_ft, criterion, MSEdis, optimizer, scheduler, num_epochs=600)
######################################
