import os

import numpy as np
from PIL import Image


def main():
    path = input("Path to npy files?")
    export_path = input("Export path?")

    npy_to_png(path, export_path)


def npy_to_png(path, export_path):
    input_path_format = os.path.join(path, 'session-{}.npy')
    output_path_img_format = os.path.join(export_path, 'features', 'img-{}.png')

    img_num = 0
    file_num = 1
    data_file = input_path_format.format(file_num)
    labels = []
    labels_file = os.path.join(export_path, 'labels', 'labels.npy')

    while os.path.isfile(data_file):
        data_set = np.load(data_file)
        data_file = input_path_format.format(file_num)

        for data_point in data_set:
            # img_file = output_path_img_format.format(img_num)
            img_num += 1

            # image_file = Image.fromarray(data_point[0])
            # image_file = image_file.convert('L')
            # image_file.save(img_file)

            labels.append(data_point[1][1])

        print(data_file)
        file_num += 1

    np.save(labels_file, np.array(labels))


if __name__ == '__main__':
    main()
