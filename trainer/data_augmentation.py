"""
Source: https://towardsdatascience.com/teaching-cars-to-drive-using-deep-learning-steering-angle-prediction-5773154608f2
"""
import cv2
import numpy as np
from PIL import Image


def h_flip_image(img):
    return cv2.flip(img, 1)


def change_image_brightness_bw(img, s_low=0.2, s_high=0.75):
    img = img.astype(np.float32)
    s = np.random.uniform(s_low, s_high)
    img[:,:] *= s
    np.clip(img, 0, 255)
    return img.astype(np.uint8)


def add_random_shadow(img, w_low=0.6, w_high=0.85):
    cols, rows = (img.shape[0], img.shape[1])

    top_y = np.random.random_sample() * rows
    bottom_y = np.random.random_sample() * rows
    bottom_y_right = bottom_y + np.random.random_sample() * (rows - bottom_y)
    top_y_right = top_y + np.random.random_sample() * (rows - top_y)
    if np.random.random_sample() <= 0.5:
        bottom_y_right = bottom_y - np.random.random_sample() * (bottom_y)
        top_y_right = top_y - np.random.random_sample() * (top_y)

    poly = np.asarray([[[top_y, 0], [bottom_y, cols], [bottom_y_right, cols], [top_y_right, 0]]], dtype=np.int32)

    mask_weight = np.random.uniform(w_low, w_high)
    origin_weight = 1 - mask_weight

    mask = np.copy(img).astype(np.int32)
    cv2.fillPoly(mask, poly, (0, 0, 0))
    # masked_image = cv2.bitwise_and(img, mask)

    return cv2.addWeighted(img.astype(np.int32), origin_weight, mask, mask_weight, 0).astype(np.uint8)


def translate_image(img, st_angle, low_x_range, high_x_range, low_y_range, high_y_range, delta_st_angle_per_px):
    rows, cols = (img.shape[0], img.shape[1])
    translation_x = np.random.randint(low_x_range, high_x_range)
    translation_y = np.random.randint(low_y_range, high_y_range)

    st_angle += translation_x * delta_st_angle_per_px
    translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    img = cv2.warpAffine(img, translation_matrix, (cols, rows))

    return img, st_angle


def augment_image(img, st_angle, p=1.0):
    aug_img = img

    if np.random.random_sample() <= p:
        aug_img = h_flip_image(aug_img)
        st_angle = -st_angle

    if np.random.random_sample() <= p:
        aug_img = change_image_brightness_bw(aug_img)

    if np.random.random_sample() <= p:
        aug_img = add_random_shadow(aug_img, w_low=0.45)

    if np.random.random_sample() <= p:
        aug_img, st_angle = translate_image(aug_img, st_angle, -60, 61, -20, 21, 0.35 / 100.0)

    aug_img = np.reshape(aug_img, (270, 480, 1))

    return aug_img, st_angle


IMG_PATH = r'D:\Documents\School work\Cegep\Session 6\EuroTruck-ai.py\data-png\features\img-{}.png'


def generate_images(ds, target_dimensions, batch_size=100,
                    data_aug_pct=0.8, aug_likelihood=0.5, st_angle_threshold=0.05, neutral_drop_pct=0.25):
    """
    Generates images whose paths and steering angle are stored in supplied dataframe object df
    Returns the tuple (batch,steering_angles)
    """
    # e.g. 160x320x3 for target_dimensions
    ds_len = len(ds)
    t_y, t_x, t_z = target_dimensions
    images = np.zeros((batch_size, t_y, t_x, t_z))
    steering_angles = np.zeros(batch_size)

    while True:
        k = 0
        while k < batch_size:
            ds_index = np.random.randint(0, ds_len)

            (im_index, st_angle) = ds[ds_index]

            if abs(st_angle) < st_angle_threshold and np.random.random_sample() <= neutral_drop_pct:
                continue

            im_frame = Image.open(IMG_PATH.format(im_index))
            im_data = np.array(im_frame)

            im_data, st_angle = \
                augment_image(im_data, st_angle, p=aug_likelihood) \
                if np.random.random_sample() <= data_aug_pct \
                else (im_data, st_angle)

            im_data = np.resize(im_data, (t_y, t_x, t_z))

            images[k] = im_data
            steering_angles[k] = st_angle

            k += 1

        yield images, np.clip(steering_angles, -1, 1)
