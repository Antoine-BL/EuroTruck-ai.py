import os

import numpy as np
from PIL import Image


def main():
    plot_all()


def plot_all():
    file_num = 100
    path = os.path.join(os.path.realpath('..\\data\\unsorted'), 'session-{}.npy')

    data_file = path.format(file_num)
    data = np.load(data_file)

    average_img_3(data)


def average_img_3(data):
    # Alternative method using numpy mean function
    pos_x = 128
    pos_y = 147

    dim_x = 200
    dim_y = 66

    images = np.array([dpoint[0] for dpoint in data])
    arr = np.array(np.mean(images, axis=0), dtype=np.uint8)
    arr = arr[pos_y:pos_y + dim_y, pos_x:pos_x + dim_x]
    out = Image.fromarray(arr)
    out.save('Average.png')


if __name__ == "__main__":
    main()
