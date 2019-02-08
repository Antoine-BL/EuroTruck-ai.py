import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    plot_folder_data('data\\unsorted', 'pre-sorted inputs')
    plot_folder_data('data\\train', 'training inputs')
    plot_folder_data('data\\test', 'validation inputs')


def plot_folder_data(rel_path, title):
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

    plt.hist(input_data, np.arange(-1, 1, 0.05))
    plt.title(title)
    plt.xlabel('Occurrences')
    plt.ylabel('steering angle')
    plt.show()


if __name__ == "__main__":
    main()
