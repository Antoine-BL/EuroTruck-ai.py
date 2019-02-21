import os
from random import random

import numpy as np

PCT_TEST = 0.2


def main():
    split_data()


def split_data():
    data_path = os.path.realpath('..\\data')
    unsorted_path = os.path.join(data_path, 'balanced', 'session-{}.npy')
    train_path = os.path.join(data_path, 'train', 'session-{}.npy')
    test_path = os.path.join(data_path, 'test', 'session-{}.npy')
    train_data = []
    test_data = []
    train_set = []
    test_set = []

    read_file_num = 1
    write_file_num = 1

    data_file = unsorted_path.format(read_file_num)
    while os.path.isfile(data_file):
        read_file_num += 1
        data_set = np.load(data_file)
        print(read_file_num)
        for data_point in data_set:
            if random() <= PCT_TEST:
                test_data.append(data_point)
                if len(test_data) >= 500:
                    np.save(test_path.format(write_file_num), test_data)
                    test_data = []
                    test_set.append(write_file_num)
                    write_file_num += 1
            else:
                train_data.append(data_point)
                if len(train_data) >= 500:
                    np.save(train_path.format(write_file_num), train_data)
                    train_data = []
                    train_set.append(write_file_num)
                    write_file_num += 1

        data_file = unsorted_path.format(read_file_num)

    np.save(test_path.format(write_file_num), test_data)
    test_set.append(write_file_num)

    write_file_num += 1
    np.save(train_path.format(write_file_num), train_data)
    train_set.append(write_file_num)

    np.save(os.path.join(data_path, 'test-set.npy'), test_set)
    np.save(os.path.join(data_path, 'train-set.npy'), train_set)


if __name__ == '__main__':
    main()
