import os
from heapq import nsmallest
import random

import numpy as np

PCT_TEST = 0.2
SAMPLES_PER_FILE = 500


def main():
    balance_data()


def balance_data():
    data_path = os.path.realpath('..\\data')
    unsorted_path = os.path.join(data_path, 'unsorted', 'session-{}.npy')
    balanced_path = os.path.join(data_path, 'balanced', 'session-{}.npy')

    dataset_size = calc_nb_samples(unsorted_path)
    print('Balancing dataset of {} samples'.format(dataset_size))

    pct_per_bin = proportions_per_bin(unsorted_path, 0.1, dataset_size)

    balance_and_save(pct_per_bin, unsorted_path, dataset_size, 0.1, balanced_path)


def calc_nb_samples(path) -> int:
    nb_files = 0

    data_file = path.format(nb_files + 1)
    while os.path.isfile(data_file):
        nb_files += 1
        data_file = path.format(nb_files + 1)

    return nb_files * SAMPLES_PER_FILE


def proportions_per_bin(path, bin_size, total_nb_samples):
    bins = np.zeros((round(2/bin_size), ), dtype=np.int)

    print('finding proportions per bin')
    nb_files = round(total_nb_samples / 500)
    for num_file in range(1, nb_files + 1):
        print('File {} of {} ({}%)'.format(num_file, nb_files, round(num_file / nb_files * 100, 1)))
        data_file = path.format(num_file)

        data = np.load(data_file)
        for data_point in data:
            bin_nb = int(round((data_point[1][1] + 1) / bin_size, 0))
            bins[bin_nb - 1] += 1

    return bins


def balance_and_save(bins, path, total_nb_samples, bin_size, write_path):
    write_file_num = 1

    second_smallest = max(nsmallest(4, bins))
    bin_prob = []
    for nb in bins:
        bin_prob.append(second_smallest / nb)

    bal_data = []
    print('Balancing data')
    nb_files = round(total_nb_samples / 500)
    for num_file in range(1, nb_files + 1):
        print('File {} of {} ({}%)'.format(num_file, nb_files, round(num_file / nb_files * 100, 1)))
        data_file = path.format(num_file)

        data = np.load(data_file)
        for data_point in data:
            bin_nb = int(round((data_point[1][1] + 1) / bin_size))

            if random.randrange(0, 10000) / 10000 < bin_prob[bin_nb - 1]:
                bal_data.append(data_point)

                if len(bal_data) == 500:
                    np.save(write_path.format(write_file_num), bal_data)
                    print('writing balanced data to file number {}'.format(write_file_num))
                    write_file_num += 1
                    bal_data = []


if __name__ == '__main__':
    main()
