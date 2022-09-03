# coding: utf-8

import os
import numpy as np
import scipy.io as scio
import pickle

def loadsubject_data(data_path):
    train_data = np.empty((40091, 100, 150))
    test_data = np.empty((16487, 100, 150))
    train_label = np.empty((40091, ), dtype='uint8')
    test_label = np.empty((16487, ), dtype='uint8')
    label = np.empty((56578, ), dtype='uint8')
    files = os.listdir(data_path)
    num = len(files)


    train_num = 0
    test_num = 0

    '''
    for i in range(num):

        one_data = scio.loadmat(os.path.join(data_path, files[i]))
        one_data = one_data['skeleton']

        # get the label
        temp = files[i][17:20]
        if list(temp)[0] == 0 and list(temp)[1] == 0:
            label[i] = int(list(temp)[2]) - 1
        else:
            label[i] = int(list(temp)[1] + list(temp)[2]) - 1


        temp1 = files[i][9:12]
        subject_num = int(list(temp1)[0] + list(temp1)[1] + list(temp1)[2])

        if subject_num == 1 or subject_num == 2 or subject_num == 4 or subject_num == 5 or subject_num == 8 or subject_num == 9\
                or subject_num == 13 or subject_num == 14 or subject_num == 15 or subject_num == 16 or subject_num == 17 \
                or subject_num == 18 or subject_num == 19 or subject_num == 25 or subject_num == 27 or subject_num == 28 \
                or subject_num == 31 or subject_num == 34 or subject_num == 35 or subject_num == 38:
            train_data[train_num, :, :] = one_data
            train_label[train_num] = label[i]
            if train_num < 40091:
                train_num = train_num + 1
                if train_num%10000==0:
                    print('train_num = ', train_num)
        else:
            test_data[test_num, :, :] = one_data
            test_label[test_num] = label[i]
            if test_num < 16487:
                test_num = test_num + 1
                if test_num%5000==0:
                    print('test_num = ', test_num)
    '''
    train_data  = np.load(data_path+"/xsub/train_data_joint.npy")
    #train_label = np.load(data_path+"/xsub/train_label.pkl")

    test_data   = np.load(data_path+"/xsub/val_data_joint.npy")
    train_file = open(data_path+"/xsub/train_label.pkl",'rb')
    train_label = np.array(pickle.load(train_file,encoding='iso-8859-1')[1])
    train_file.close
    test_file = open(data_path+"/xsub/val_label.pkl",'rb')
    file = pickle.load(test_file,encoding='iso-8859-1')
    ttt,test_label = np.array(file[0]),np.array(file[1])
    #print(ttt[0],test_label[0])
    #print(train_data[0][0])
    test_file.close
    #test_label  = np.load(data_path+"/xsub/val_label.pkl")
    return train_data, train_label, test_data, test_label


def loadview_data(data_path):
    train_data = np.empty((37646, 100, 150))
    test_data = np.empty((18932, 100, 150))
    train_label = np.empty((37646, ), dtype='uint8')
    test_label = np.empty((18932, ), dtype='uint8')
    label = np.empty((56578, ), dtype='uint8')
    files = os.listdir(data_path)
    num = len(files)

    train_num = 0
    test_num = 0

    '''
    for i in range(num):
        one_data = scio.loadmat(os.path.join(data_path, files[i]))
        one_data = one_data['skeleton']

        # get the label
        temp = files[i][17:20]
        if list(temp)[0] == 0 and list(temp)[1] == 0:
            label[i] = int(list(temp)[2]) - 1
        else:
            label[i] = int(list(temp)[1] + list(temp)[2]) - 1

        temp1 = files[i][5:8]
        view_num = int(list(temp1)[0] + list(temp1)[1] + list(temp1)[2])


        if view_num == 2 or view_num == 3:
            train_data[train_num, :, :] = one_data
            train_label[train_num] = label[i]
            if train_num < 37646:
                train_num = train_num + 1
                if train_num%10000==0:
                    print('train_num = ', train_num)
        else:
            test_data[test_num,:,:] = one_data
            test_label[test_num] = label[i]
            if test_num < 18932:
                test_num = test_num + 1
                if test_num % 5000 == 0:
                    print('test_num = ', test_num)
    '''

    train_data  = np.load(data_path+"/xview/train_data_joint.npy")
    #train_label = np.load(data_path+"/xsub/train_label.pkl")

    test_data   = np.load(data_path+"/xview/val_data_joint.npy")
    train_file = open(data_path+"/xview/train_label.pkl",'rb')
    train_label = np.array(pickle.load(train_file,encoding='iso-8859-1')[1])
    train_file.close
    test_file = open(data_path+"/xview/val_label.pkl",'rb')
    file = pickle.load(test_file,encoding='iso-8859-1')
    ttt,test_label = np.array(file[0]),np.array(file[1])
    test_file.close
    return train_data, train_label, test_data, test_label


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = loadview_data()
