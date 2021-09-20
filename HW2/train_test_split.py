"""
SPDKH
09/20/2021
"""
# # Creating Train / Val / Test folders (One time use)
import os
import shutil
import numpy as np


def train_test_split(root_dir='15-Scene Image Dataset/15-Scene',
                     classes_dir=['00', '01', '02', '03', '04',
                                  '05', '06', '07', '08', '09',
                                  '10', '11', '12', '13', '14'],
                     test_ratio=0.3):
    """
    To split train and test sets saved in root_dir
    each in a class dir given as classes_dir
    with the test ratio given as float
    """
    for i in classes_dir:
        if not os.path.exists(root_dir + '/train/' + i):
            os.makedirs(root_dir + '/train/' + i)

            os.makedirs(root_dir + '/test/' + i)

            source = root_dir + '/' + i
            all_file_names = os.listdir(source)
            np.random.shuffle(all_file_names)
            train_file_names, test_file_names = np.split(np.array(all_file_names),
                                                         [int(len(all_file_names) *\
                                                              (1 - test_ratio))])

            train_file_names = [source + '/' + name for name in train_file_names.tolist()]
            test_file_names = [source + '/' + name for name in test_file_names.tolist()]

            for name in train_file_names:
                shutil.copy(name, root_dir + '/train/' + i)

            for name in test_file_names:
                shutil.copy(name, root_dir + '/test/' + i)
