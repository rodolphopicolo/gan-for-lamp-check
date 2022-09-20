#!/usr/bin/env python

__author__ = 'Rodolpho Picolo'
__email__ = 'rodolphopicolo@gmail.com'

'''
  Check all images of the dataset are of the same size.
  All images are of shape (480, 640, 3), if would have difference among images
  we would have to set all of then with the same size.
'''
import cv2 as cv
import sys
import os


def dataset_dir():
    if len(sys.argv) < 2:
        raise Exception('Dataset dir path not specified')

    dataset_dir_path = sys.argv[1]
    if not os.path.isdir(dataset_dir_path):
        raise Exception('Dataset dir path is not a directory')

    return dataset_dir_path


def dataset_file_list(dataset_dir_path):
    dataset_files = os.listdir(dataset_dir_path)
    dataset_files = [os.path.join(dataset_dir_path, f) for f in dataset_files]    
    return dataset_files


def check_dataset_images_shapes(dataset_files):
    shapes = {}
    for image_path in dataset_files:
        img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        print(image_path, 'shape:', img.shape, 'min_value:', img.min(), 'max_value:', img.max(), 'size:', img.size)
        shape = img.shape
        if not shape in shapes:
            shapes[shape] = 1
        else:
            shapes[shape] += 1
    print(shapes)


def main():
    dataset_dir_path = dataset_dir()
    dataset_files = dataset_file_list(dataset_dir_path)
    check_dataset_images_shapes(dataset_files)


if __name__ == '__main__':
    main()