import sys
import os
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
#from images_helper import generate_and_save_images, check_previous_epochs, convert_generated_images_to_visible_mode

MNIST_IMAGE_SHAPE = (28, 28, 1)
LAMPS_IMAGE_SHAPE = (480, 640, 3)

def load_parameters():
    if len(sys.argv) < 3:
        raise Exception("Dataset dir path and dataset index file path must be specified as command line parameters")
    dataset_dir_path = sys.argv[1]
    dataset_metadata_file_path = sys.argv[2]
    return dataset_dir_path, dataset_metadata_file_path

def load_dataset_metadata(dataset_metadata_file_path):
    dataset_metadata = {}
    index_df = pd.read_csv(dataset_metadata_file_path);
    lamp_index = 0
    material_index = 1
    power_index = 2
    first_photo_index = 3
    last_photo_index = 4
    for df_index, row in index_df.iterrows():
        lamp = int(row[lamp_index])
        material = row[material_index]
        power = int(row[power_index])
        first_photo = int(row[first_photo_index])
        last_photo = int(row[last_photo_index])

        for i in range(first_photo, last_photo + 1):
            photo_metadata = {
                'label':lamp,
                'lamp':lamp,
                'material': material,
                'power': power,
                'first_photo': first_photo,
                'last_photo': last_photo,
                'photo': i
            }

            dataset_metadata[i] = photo_metadata

    return dataset_metadata




def load_lamps_dataset(dataset_dir_path, dataset_metadata_file_path):
    if dataset_dir_path == None or dataset_metadata_file_path == None:
        dataset_dir_path, dataset_metadata_file_path = load_parameters()

    dataset_metadata = load_dataset_metadata(dataset_metadata_file_path)
    list_metadata = []
    images = []

    dataset_files = os.listdir(dataset_dir_path)
    dataset_files.sort()
    index = 0
    for file_name in dataset_files:
        file_path = os.path.join(dataset_dir_path, file_name)

        img = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        img = (img - 127.5) / 127.5

        if len(images) == 0:
            images = np.zeros(((len(dataset_metadata),) + img.shape))
        images[index] = img


        photo_id = int(file_name[5: -4])
        sample = dataset_metadata[photo_id]
        sample['file_path'] = file_path
        sample['file_name'] = file_name
        sample['img'] = img

        list_metadata.append(sample)

        index += 1

    return images, dataset_metadata, list_metadata

def load_mnist_dataset():
    import tensorflow as tf
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return train_images


def save_image_minus_one_to_one(image, path):
    #image_rgb = convert_generated_images_to_visible_mode(image)
    image_bgr = ((image * 127.5) + 127.5).astype(dtype=np.uint8)

    image_rgb = np.zeros(image_bgr.shape, dtype=np.uint8)

    image_rgb[:,:,0] = image_bgr[:,:,2]
    image_rgb[:,:,1] = image_bgr[:,:,1]
    image_rgb[:,:,2] = image_bgr[:,:,0]
   
    plt.imsave(path, image_rgb)

def main():
    lamps, metadata, list_metadata = load_lamps_dataset('../fotos_28_07', '../lamp-index.csv')
    #mnist = load_mnist_dataset()
    for metadata in list_metadata:
        image = metadata['img']
        name = metadata['file_name']
        path = '/tmp/gan_images/' + name
        save_image_minus_one_to_one(image, path)

    

if __name__ == '__main__':
    lamps = main()
    