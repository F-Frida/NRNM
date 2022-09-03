import os
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import pickle

def vids_with_missing_skeletons():
    f = open('samples_with_missing_skeletons.txt', 'r')
    bad_files = []
    for line in f:
        bad_files.append(line.strip()+'.skeleton')
    f.close()
    return bad_files

class save_skeleton_data_from_NTU():
    def __init__(self):
        self.skeletondata_path = 'D:/NTU/nturgbd_skeletons/nturgb+d_skeletons/'
        self.save_path = 'D:/NTU/nturgbd_skeletons/single/'
        data_list = os.listdir(self.skeletondata_path)
        bad_files = vids_with_missing_skeletons()  # get the missing skeleton
        for i, data_file in enumerate(data_list):
            if data_file in bad_files:
                continue
            data = self.load_skeleton_data(data_file)
            save_file = data_file.split('.')[0] + '.pk'
            with open(self.save_path + save_file, 'wb') as f:
                pickle.dump(data, f)
            del data

    def load_skeleton_data(self, file_name):
        truthlabel = (int(file_name.split('.')[0].split('A')[1]))
        file = pd.read_table(self.skeletondata_path + file_name, header=None)
        file_iters = iter(file[0])

        frame_count = int(next(file_iters))
        skeleton_data = []
        for f in range(frame_count):
            body_count = int(next(file_iters))
            body_info = []
            for j in range(body_count):
                body = {}
                info_text = next(file_iters).split(' ')
                body['bodyID'] = info_text[0]
                body['clippedEdges'] = info_text[1]
                body['handLeftConfidence'] = info_text[2]
                body['handLeftState'] = info_text[3]
                body['handRightConfidence'] = info_text[4]
                body['handRightState'] = info_text[5]
                body['isRestricted'] = info_text[6]
                body['leanX'] = info_text[7]
                body['leanY'] = info_text[8]
                body['trackingState'] = info_text[9]
                body['jointCount'] = int(next(file_iters))
                joints = {}
                for t in range(body['jointCount']):    # we only use x,y,z
                    joint = {}
                    info_text = next(file_iters).split(' ')
                    joint['x'] = float(info_text[0])
                    joint['y'] = float(info_text[1])
                    joint['z'] = float(info_text[2])
                    joint['depthX'] = float(info_text[3])
                    joint['depthY'] = float(info_text[4])
                    joint['colorX'] = float(info_text[5])
                    joint['colorY'] = float(info_text[6])
                    joint['orientationW'] = float(info_text[7])
                    joint['orientationX'] = float(info_text[8])
                    joint['orientationY'] = float(info_text[9])
                    joint['orientationZ'] = float(info_text[10])
                    joint['trackingState'] = info_text[11]
                    joints[t] = joint
                body['joints'] = joints
                body_info.append(body.copy())
                body.clear()
            skeleton_data.append(body_info.copy())
            body_info.clear()
        return skeleton_data, truthlabel


class read_skeleton_file_from_NTU():
    def __init__(self, save_path='', train=True, cross_subject=True, cross_view=False):
        self.save_path = save_path
        self.train = train
        self.cross_subject = cross_subject
        self.cross_view = cross_view
        self.cross_subject_train = []
        self.cross_subject_test = []
        self.cross_view_train = []
        self.cross_view_test = []
        self.cross_subject_train_index = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        self.cross_subject_test_index = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]
        self.cross_view_train_index = [2, 3]
        self.cross_view_test_index = [1]

        file_list = os.listdir(self.save_path)
        for i, file_name in enumerate(file_list):
            f_name = os.path.splitext(file_name)[0]
            self.load_cross_subject_data(f_name)
            self.load_cross_view_data(f_name)

    def load_cross_subject_data(self, file_name):
        subject_index = int(file_name.split('P')[1].split('R')[0])
        if subject_index in self.cross_subject_train_index:
            print('load cross subject train data %s' % file_name)
            # self.cross_subject_train.append(self.save_path + file_name + '.pk')
            self.cross_subject_train.append(file_name + '.pk')
        elif subject_index in self.cross_subject_test_index:
            print('load cross subject test data %s' % file_name)
            # self.cross_subject_test.append(self.save_path + file_name + '.pk')
            self.cross_subject_test.append(file_name + '.pk')

    def load_cross_view_data(self, file_name):
        view_index = int(file_name.split('C')[1].split('P')[0])
        if view_index in self.cross_view_train_index:
            print('load cross view train data %s' % file_name)
            # self.cross_view_train.append(self.save_path + file_name + '.pk')
            self.cross_view_train.append(file_name + '.pk')
        elif view_index in self.cross_view_test_index:
            print('load cross view test data %s' % file_name)
            # self.cross_view_test.append(self.save_path + file_name + '.pk')
            self.cross_view_test.append(file_name + '.pk')


class NTUDataSet(data.Dataset):
    def __init__(self, data_root='', train = True, cross_subject = True, cross_view = False):
        super(NTUDataSet, self).__init__()
        # with open('./ntu_file.pk', 'rb') as f:
        #     self.data = pickle.load(f)
        self.data = read_skeleton_file_from_NTU(save_path=data_root)
        self.data.train = train
        self.data.cross_subject = cross_subject
        self.data.cross_view = cross_view
        self.data_root = data_root

    def normalize(self, joints):
        pass

    def __getitem__(self, index):
        if self.data.train:
            if self.data.cross_subject:
                file_name = self.data.cross_subject_train[index]
            elif self.data.cross_view:
                file_name = self.data.cross_view_train[index]
        else:
            if self.data.cross_subject:
                file_name = self.data.cross_subject_test[index]
            elif self.data.cross_view:
                file_name = self.data.cross_view_test[index]

        with open(os.path.join(self.data_root, file_name), 'rb') as f:
            data = pickle.load(f)
        skeleton_data, target = data

        frame_count = len(skeleton_data)    # select frames
        clip_length = int(np.rint(frame_count / 8))
        chosen_frame = []
        for i in range(7):
            rand_index = np.random.randint(0, clip_length-1)
            chosen_frame.append(skeleton_data[i*clip_length + rand_index])
        rand_index = np.random.randint(7*clip_length-1, frame_count-1)
        chosen_frame.append(skeleton_data[rand_index])

        output_joints = []
        for i in range(8):   # bug !!!
            joint = np.zeros((25*3))
            for b in range(len(chosen_frame[i])):
                if chosen_frame[i][b]['trackingState'] == '0':
                    continue
                for j, value in chosen_frame[i][b]['joints'].items():
                    joint[b*3*25+3*j] = value['x']
                    joint[b*3*25+3*j+1] = value['y']
                    joint[b*3*25+3*j+2] = value['z']
                new_origin = np.array([joint[3], joint[4], joint[5]])
                new_axisx = np.array([joint[24], joint[25], joint[26]]) - np.array([joint[12], joint[13], joint[14]])
                new_axisx = new_axisx / np.linalg.norm(new_axisx, ord=2)
                new_axisy = np.array([joint[0], joint[1], joint[2]]) - np.array([joint[60], joint[61], joint[62]])
                new_axisy = new_axisy / np.linalg.norm(new_axisy, ord=2)
                new_axisz = np.cross(new_axisx, new_axisy)
                joint = normalize_rotate(new_axisx, new_axisy, new_axisz, new_origin, joint)
            output_joints.append(joint)
        return torch.Tensor(output_joints), target

    def __len__(self):
        if self.data.train:
            if self.data.cross_subject:
                return len(self.data.cross_subject_train)
            elif self.data.cross_view:
                return len(self.data.cross_view_train)
        else:
            if self.data.cross_subject:
                return len(self.data.cross_subject_test)
            elif self.data.cross_view:
                return len(self.data.cross_view_test)


def normalize_rotate(new_axisx, new_axisy, new_axisz, new_origin, input):
    output = np.zeros(input.shape)
    for i in range(24):
        vec = input[i*3:(i+1)*3] - new_origin
        output[i*3] = np.dot(vec, new_axisx.T) / np.sqrt(new_axisx.dot(new_axisx))
        output[i*3+1] = np.dot(vec, new_axisy.T) / np.sqrt(new_axisy.dot(new_axisy))
        output[i*3+2] = np.dot(vec, new_axisz.T) / np.sqrt(new_axisz.dot(new_axisz))
    return output


if __name__ == "__main__":
    convert_NTU = save_skeleton_data_from_NTU()   # convert to pickle
    save_file = 'D:/NTU/nturgbd_skeletons/ntu_file.pk'
    read_ntu = read_skeleton_file_from_NTU()      # save index
    with open(save_file, 'wb') as f:
        pickle.dump(read_ntu, f)
    pkl_file = open(save_file, 'rb')
    test = pickle.load(pkl_file)
    pkl_file.close()

    print(test.cross_subject_test_index)
    print('down')

