import gc
import os

import numpy as np


def main():
    mirror_images()


def mirror_images():
    rel_path = '..\\data\\unsorted'
    path = os.path.join(os.path.realpath(rel_path), 'session-{}.npy')

    read_file_num = 1
    write_file_num = 1

    write_data_file = path.format(write_file_num)
    while os.path.isfile(write_data_file):
        write_file_num += 1
        write_data_file = path.format(write_file_num)
    read_file_lim = write_file_num
    print(write_file_num)

    data_file = path.format(read_file_num)

    while os.path.isfile(data_file) and read_file_num < read_file_lim:
        data_set = np.load(data_file)
        mirrored_data = []
        print(read_file_num, write_file_num)

        for data_point in data_set:
            data_point[0] = np.fliplr(data_point[0])
            data_point[1][1] *= -1
            mirrored_data.append(data_point)

        np.save(write_data_file, mirrored_data)

        write_file_num += 1
        read_file_num += 1
        data_file = path.format(read_file_num)
        write_data_file = path.format(write_file_num)

        gc.collect()

if __name__ == "__main__":
    main()
