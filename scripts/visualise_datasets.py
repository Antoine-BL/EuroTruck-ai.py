import os

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = os.path.realpath('..\\data')


def main():
    plot_all()


def plot_all():
    plot_unsorted_data('..\\data\\balanced', 'balanced data')
    # plot_unsorted_data('..\\data\\unsorted', 'unsorted inputs')
    plot_training_set('..\\data\\train', 'train-set.npy', 'training inputs')
    plot_training_set('..\\data\\test', 'test-set.npy', 'validation inputs')


def plot_unsorted_data(rel_path, title):
    file_num = 1
    path = os.path.join(os.path.realpath(rel_path), 'session-{}.npy')
    input_data = []

    data_file = path.format(file_num)
    while os.path.isfile(data_file):
        file_num += 1
        data_set = np.load(data_file)
        data_file = path.format(file_num)
        print(file_num)
        for data_point in data_set:
            input_data.append(data_point[1][1])

    plt.hist(input_data, np.arange(-1, 1.05, 0.05))
    plt.title(title)
    plt.xlabel('steering angle')
    plt.ylabel('Occurrences')
    plt.show()


def plot_training_set(rel_path, set_file, title):
    path = os.path.join(os.path.realpath(rel_path), 'session-{}.npy')
    set_path = os.path.join(DATA_PATH, set_file)
    set = np.load(set_path)
    input_data = []

    for file_num in set:
        data_file = path.format(file_num)
        data_set = np.load(data_file)
        print(file_num)
        for data_point in data_set:
            input_data.append(data_point[1][1])

    plt.hist(input_data, np.arange(-1, 1.05, 0.05))
    plt.title(title)
    plt.xlabel('steering angle')
    plt.ylabel('Occurrences')
    plt.show()


if __name__ == "__main__":
    main()
