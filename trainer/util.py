import os

import tensorflow as tf

from tensorflow.python.keras.preprocessing import image

from tensorflow.python.keras.backend import resize_images
import cv2
import numpy as np

JOY_INDEX = 1
INPUT_INDEX = 1
IMG_INDEX = 1
PCT_VALIDATION = 0.2
PCT_TRAIN = 1 - PCT_VALIDATION
LABELS_PATH = 'data-png/labels/labels.npy'
IMGS_PATH = 'data-png/features/img-{}.png'
MODELS_PATH = 'models'
CPOINT_PATH = 'models/checkpoints/weights{epoch:02d}-{val_loss:.4f}.h5'


def resize(img):
    return resize_images(img, 66, 200, 'channels_first')


def normalize(img):
    normalized = img
    return (img / 255.0) - 0.5


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


def tf_load_png_data(uri, sess):
    img = image.load_img(uri, color_mode='grayscale')
    image_array = image.img_to_array(img)
    return image_array
