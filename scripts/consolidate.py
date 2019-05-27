import os

import numpy as np


def main():
    path = input("Path to files?")
    max_file_nb = int(input("Max file nb?"))

    consolidate_files(path, max_file_nb)


def consolidate_files(path, max_file_nb):
    input_path_format = os.path.join(path, 'img-{}.png')
    output_path_img_format = os.path.join(path, 'img-{}.png')
    file_nb = 0

    labels = np.load('D:\\Documents\\School work\\Cegep\\Session 6\\EuroTruck-ai.py\\data-png\\labels\\labels.npy')

    for i in range(0, max_file_nb + 1):
        in_file = input_path_format.format(i)

        if os.path.isfile(in_file):
            out_file = output_path_img_format.format(file_nb)
            file_nb += 1
            os.rename(in_file, out_file)
        else:
            labels = np.delete(labels, file_nb)

    np.save('D:\\Documents\\School work\\Cegep\\Session 6\\EuroTruck-ai.py\\data-png\\labels\\labels.npy', labels)

if __name__ == '__main__':
    main()
