import cv2
import os

import numpy as np
from PIL import Image


def main():
    average_img_3()


def average_img_3():
    # Alternative method using numpy mean function
    dim_x = 280
    dim_y = 100
    pos_x = 100
    pos_y = 130
    samples_per_file = 500

    path = '..\\data\\balanced\\session-{}.npy'

    file_count = 1
    while os.path.isfile(path.format(file_count)):
        file_count += 1

    write_file_num = 1

    num_samples = file_count * samples_per_file
    im_mean = np.zeros(shape=(270, 480), dtype='float32')
    write_data_file = path.format(write_file_num)
    while os.path.isfile(write_data_file):
        print(write_file_num)
        data = np.load(write_data_file)

        for data_point in data:
            im_data = data_point[0]
            # im_data = im_data[pos_y:pos_y + dim_y, pos_x:pos_x + dim_x]
            im_mean = np.add(im_mean, np.array(im_data / num_samples))

        write_file_num += 1
        write_data_file = path.format(write_file_num)

    out = Image.fromarray(im_mean)
    out.show('Average')

    im_example = np.load(path.format(write_file_num - 1))[0][0]
    np.flipud(im_example)
    out2 = Image.fromarray(im_example - im_mean)
    out2.show('Minus mean')


if __name__ == "__main__":
    main()
