import os
import random
import numpy as np
import scipy as sp


def read(dir, type, res, **kwargs):
    # Get keyword arguments and their defaults
    augment = kwargs.get('augment', False)
    augment_level = kwargs.get('level', 6)
    augment_ops = kwargs.get('ops', [flip_xy, flip_xz, flip_xz])

    # Find the list of files
    voxel_path = dir + "/" + type + "/" + "inouts" + str(res)
    voxel_files = sorted(os.listdir(voxel_path))

    # Try to add some randomization
    random.shuffle(voxel_files)

    all_data = []
    for idx, file in enumerate(voxel_files):
        # Read raw file using NumPy
        filedata_pre = np.fromfile(voxel_path + "/" + file, dtype='uint8')
        # Reshape it into the desired resolution
        filedata = np.reshape(filedata_pre, (res, res, res))
        # Add to the return list
        if type == 'train':
            new_file = sp.ndimage.rotate(filedata, 50, axes=(1, 0), order = 0, reshape=False, mode='constant')
            all_data.append(new_file)
        else:
            all_data.append(filedata)

        # A simple data augmentation
        if augment:
            # Effect of augmentation level
            if idx % augment_level == 0:
                # Apply chosen data augmentation methods
                for ops in augment_ops:
                    filedata_aug = ops(filedata)
                    all_data.append(filedata_aug)

    return all_data


def flip_xy(data):
    temp = np.empty_like(data)
    temp[:, [0, 1]] = data[:, [1, 0]]
    return temp


def flip_xz(data):
    temp = np.empty_like(data)
    temp[:, [0, -1]] = data[:, [-1, 0]]
    return temp


def flip_yz(data):
    temp = np.empty_like(data)
    temp[:, [1, 2]] = data[:, [2, 1]]
    return temp


def create_input(data_path, res):
    data_files = sorted(os.listdir(data_path))
    num_classes = np.arange(len(data_files))
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    fileid = 0
    for files in data_files:
        temp_path = data_path + "/" + files
        temp_train = read(temp_path, 'train', res, augment=True, level=1)
        temp_test = read(temp_path, 'test', res, augment=True, level=1)
        Y_train.append([num_classes[fileid]]*len(temp_train))
        Y_test.append([num_classes[fileid]]*len(temp_test))
        X_train.append(temp_train)
        X_test.append(temp_test)
        fileid = fileid + 1
    X_train = [s for train in X_train for s in train]
    X_test = [s for test in X_test for s in test]
    Y_train = [s for train in Y_train for s in train]
    Y_test = [s for test in Y_test for s in test]
    return (X_train, Y_train), (X_test, Y_test), num_classes.shape[0]
