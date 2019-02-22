import os
from collections import Iterable
from os import path

import numpy as np


def main():
    nb_passes = 1

    for x in range(0, nb_passes):
        file_nums = get_file_nums()
        shuffle_files(file_nums)


def get_file_nums() -> list:
    file_num = 1
    rel_path = '..\\data\\balanced'
    path = os.path.join(os.path.realpath(rel_path), 'session-{}.npy')
    data_file = path.format(file_num)
    file_nums = []

    while os.path.isfile(data_file):
        file_nums.append(file_num)
        file_num += 1
        data_file = path.format(file_num)


    arr = np.array(file_nums)
    np.random.shuffle(arr)

    return list(arr.tolist())


def shuffle_files(file_nums: list):
    rel_path = '..\\data\\balanced'
    rel_path_dest = '..\\data\\shuffled'
    path = os.path.join(os.path.realpath(rel_path), 'session-{}.npy')
    output_path = os.path.join(os.path.realpath(rel_path_dest), 'session-{}.npy')
    total_files = len(file_nums)

    write_file_num = 1

    print('shuffling {} files'.format(total_files))
    for i in range(0, len(file_nums) - 1, 2):
        file_1 = path.format(file_nums[i])
        file_2 = path.format(file_nums[i + 1])

        all_data = np.concatenate([np.load(file_1), np.load(file_2)])
        np.random.shuffle(all_data)

        write_file = output_path.format(write_file_num)
        np.save(write_file, all_data[:500])
        write_file_num += 1

        write_file = output_path.format(write_file_num)
        np.save(write_file, all_data[-500:])
        write_file_num += 1

        print('File {} of {} ({}%)'.format(write_file_num, total_files, round(write_file_num / total_files * 100, 1)))


if __name__ == "__main__":
    main()
