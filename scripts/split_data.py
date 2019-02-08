import os
from random import random

import numpy as np

PCT_TEST = 0.2

def main():
    path = os.path.join(os.path.realpath('data\\unsorted'), 'session-{}.npy')
    train_path = os.path.join(os.path.realpath('data\\train'), 'session-{}.npy')
    test_path = os.path.join(os.path.realpath('data\\test'), 'session-{}.npy')
    train_data = []
    test_data = []

    file_num = 1
    train_file_num = 1
    test_file_num = 1

    data_file = path.format(file_num)
    while os.path.isfile(data_file):
        file_num += 1
        data_set = np.load(data_file)
        print(file_num)
        for data_point in data_set:
            if random() <= PCT_TEST:
                test_data.append(data_point)
                if len(test_data) % 500 == 0:
                    np.save(test_path.format(test_file_num), test_data)
                    test_data = []
                    test_file_num += 1
            else:
                train_data.append(data_point)
                if len(train_data) % 500 == 0:
                    np.save(train_path.format(train_file_num), train_data)
                    train_data = []
                    train_file_num += 1

        data_file = path.format(file_num)

    np.save(test_path.format(test_file_num), test_data)
    np.save(train_path.format(train_file_num), train_data)


if __name__ == '__main__':
    main()
