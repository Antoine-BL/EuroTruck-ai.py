import os
import shutil
from collections import Iterable
from os import path

import numpy as np


def main():
    shuffle_data()


def shuffle_data():
    nb_passes = 5

    src_path = '..\\data\\balanced'
    dest_path = '..\\data\\shuffle-{}'
    for n in range(0, nb_passes):
        print('Pass {} of {}'.format(n + 1, nb_passes))
        file_nums = get_file_nums()

        if n == nb_passes - 1:
            dest_path = '..\\data\\shuffled'

        os.makedirs(dest_path.format(n), exist_ok=True)
        file_nums_cp = file_nums.copy()
        np.random.shuffle(file_nums_cp)
        shuffle_files(file_nums_cp, src_path.format(n - 1), dest_path.format(n))

        src_path = dest_path

    for n in range(0, nb_passes - 1):
        shutil.rmtree('..\\data\\shuffle-{}'.format(n))


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

    return list(arr.tolist())


def shuffle_files(file_nums: list, dest_p, src_p):
    path = os.path.join(os.path.realpath(dest_p), 'session-{}.npy')
    output_path = os.path.join(os.path.realpath(src_p), 'session-{}.npy')
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

    if total_files % 2 != 0:
        print('odd nb of files, shuffling excluded one')
        last_file = path.format(file_nums[-1])
        last_data = np.load(last_file)
        np.random.shuffle(last_data)
        np.save(output_path.format(total_files), last_data)


if __name__ == "__main__":
    main()
