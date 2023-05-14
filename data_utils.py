import os
import cv2
import numpy as np

def load_training_images(data_path, n_subjects, n_images_per_subject, img_ext='.pgm', max_width=100, max_height=100):
    images = []
    labels = []

    for subject_id in range(1, n_subjects + 1):
        subject_dir = os.path.join(data_path, f's{subject_id}')
        for img_id in range(1, n_images_per_subject + 1):
            img_path = os.path.join(subject_dir, f'{img_id}{img_ext}')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img.shape[0] > max_height or img.shape[1] > max_width:
                img = cv2.resize(img, (max_width, max_height))
            images.append(img)
            labels.append(subject_id)

    return np.array(images), np.array(labels)

def load_test_images(data_path, n_subjects, img_ext='.pgm', max_width=100, max_height=100):
    images = []
    labels = []

    for subject_id in range(1, n_subjects + 1):
        subject_dir = os.path.join(data_path, f's{subject_id}')
        img_path = os.path.join(subject_dir, f'{subject_id}{img_ext}')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img.shape[0] > max_height or img.shape[1] > max_width:
            img = cv2.resize(img, (max_width, max_height))
        images.append(img)
        labels.append(subject_id)

    return np.array(images), np.array(labels)
