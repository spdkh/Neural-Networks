"""
SPDKH
09/20/2021
"""
# # Creating Train / Val / Test folders (One time use)
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras


def train_test_split(root_dir: str,
                     classes_dir=['00', '01', '02', '03', '04',
                                  '05', '06', '07', '08', '09',
                                  '10', '11', '12', '13', '14'],
                     test_ratio=0.3):
    """
    source: https://www.kaggle.com/questions-and-answers/102677
    To split train and test sets saved in root_dir
    each in a class dir given as classes_dir
    with the test ratio given as float
    """
    for i in classes_dir:
        if not os.path.exists(root_dir + 'train/' + i):
            os.makedirs(root_dir + 'train/' + i)

            os.makedirs(root_dir + 'test/' + i)

            source = root_dir + i
            all_file_names = os.listdir(source)
            np.random.shuffle(all_file_names)
            train_file_names, test_file_names = np.split(np.array(all_file_names),
                                                         [int(len(all_file_names) *\
                                                              (1 - test_ratio))])

            train_file_names = [source + '/' + name for name in train_file_names.tolist()]
            test_file_names = [source + '/' + name for name in test_file_names.tolist()]

            for name in train_file_names:
                shutil.copy(name, root_dir + 'train/' + i)

            for name in test_file_names:
                shutil.copy(name, root_dir + 'test/' + i)


# load image as pixel array
def load_images_from_folder(folder):
    """
    source: https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
    source: https://github.com/zahangircse/COMP_EECE_7or8740_NNs/blob/main/Lecture_8.ipynb

    """
    images = []
    classes = []
    for cls in os.listdir(folder):
        for filename in os.listdir(folder + cls + '/'):
            img = np.asarray(plt.imread(os.path.join(folder + cls + '/', filename)))
            if len(img.shape) > 2:
                img = np.mean(img, axis=2)
            img = img.reshape(img.shape[0], img.shape[1], 1)
            if img is not None:
                # to have known image input size
                images.append(img[:200, :200, :])
                # to augment images by cropping
                images.append(img[-200:, -200:, :])
                images.append(img[-200:, :200, :])
                images.append(img[:200, -200:, :])
                # print(img[:200, :200].shape)
                # print(img[-200:, -200:].shape)
                # print(img[-200:, :200].shape)
                # print(img[:200, -200:].shape)
                for i in range(4):
                    classes.append(cls)
    # print(np.array(images))
    return np.array(images), np.asarray(classes)


def load_data(root_dir, num_classes=15):
    # The data, split between train and test sets:
    (x_train, y_train) = load_images_from_folder(root_dir + '/train/')
    (x_test, y_test) = load_images_from_folder(root_dir + '/test/')
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # plot first few images
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(x_train[i])
    # show the figure
    plt.show()

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)

