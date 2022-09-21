# -*- coding: utf-8 -*-
# https://www.tensorflow.org/tutorials/generative/dcgan


import sys
import os
import time
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as pyplot

import tensorflow as tf
from tensorflow.keras import layers
from IPython import display


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
EPOCHS = 1
BATCH_SIZE = 1
noise_dim = 10
num_examples_to_generate = 1

IMAGE_SHAPE = (480, 640, 3)


def load_parameters():

    if len(sys.argv) < 3:
        raise Exception("Dataset dir path and dataset index file path must be specified as command line parameters")

    dataset_dir_path = sys.argv[1]
    dataset_index_file_path = sys.argv[2]

    return dataset_dir_path, dataset_index_file_path

def load_dataset_index(dataset_index_file_path):
    dataset = {}

    index_df = pd.read_csv(dataset_index_file_path);

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
            photo_spec = {
                'lamp':lamp,
                'material': material,
                'power': power,
                'first_photo': first_photo,
                'last_photo': last_photo,
                'photo': i
            }

            dataset[i] = photo_spec

    return dataset

def load_dataset(dataset_dir_path, dataset):
    dataset_files = os.listdir(dataset_dir_path)
    for file_name in dataset_files:
        file_path = os.path.join(dataset_dir_path, file_name)

        img = cv.imread(file_path, cv.IMREAD_UNCHANGED)

        photo_id = int(file_name[5: -4])
        sample = dataset[photo_id]
        sample['file_path'] = file_path
        sample['file_name'] = file_name
        sample['img'] = img

    return dataset

def create_image_and_labels_tupla(dataset):
    keys = list(dataset.keys())
    keys.sort()
    first_key = keys[0]
    shape = (len(dataset),) + dataset[first_key]['img'].shape
    train_images = np.zeros(shape, dtype=np.float32)
    train_labels = np.zeros((len(dataset)), dtype=int)

    for i in range(0, len(keys)):
        key = keys[i]
        sample = dataset.get(key)
        lamp = sample['lamp']
        img = sample['img']

        img = (img - 127.5) / 127.5

        train_images[i] = img
        train_labels[i] = lamp

    return train_images, train_labels
    

def make_generator_model():
    IMAGE_SHAPE = (480, 640, 3)
    shape_1 = (480, 640, 3)

    divisor = (10, 10, 1)
    shape_1 = tuple(np.array(tuple(np.array((480, 640, 3)) / divisor), dtype=int))

    model = tf.keras.Sequential()
    model.add(layers.Dense(np.prod(shape_1), use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    

    model.add(layers.Reshape((shape_1) + (1,)))

    output_1 = model.output_shape
    print('model.output_shape 1:', output_1)

    assert model.output_shape == (((None, ) + shape_1 + (1,)))  # Note: None is the batch size
    
    filters_1 = 128
    strides_1 = (4, 4, 1)
    kernel_size_1 = (12, 18, 3)
    model.add(layers.Conv3DTranspose(filters_1, kernel_size_1, strides=strides_1, padding='same', use_bias=False))

    output_2 = model.output_shape
    print('model.output_shape 2:', output_2)

    assert model.output_shape == ((None, ) + tuple(np.array(output_1[1:-1]) * np.array(strides_1)) + (filters_1,))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    filters_2 = 64
    strides_2 = (4, 4, 1)
    kernel_size_2 = (12, 18, 1)
    model.add(layers.Conv3DTranspose(filters_2, kernel_size_2, strides=strides_2, padding='same', use_bias=False))

    output_3 = model.output_shape
    print('model.output_shape 3:', output_3)

    assert model.output_shape == ((None, ) + tuple(np.array(output_2[1:-1]) * np.array(strides_2)) + (filters_2,))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    filters_3 = 1
    strides_3 = (4, 4, 3)
    kernel_size_3 = (6, 9, 3)
    model.add(layers.Conv3DTranspose(filters_3, kernel_size_3, strides=strides_3, padding='same', use_bias=False, activation='tanh'))
    
    output_4 = model.output_shape
    print('model.output_shape 4:', output_4)

    assert model.output_shape == ((None, ) + tuple(np.array(output_3[1:-1]) * np.array(strides_3)) + (filters_3,))


    pool_size = tuple(np.array(strides_1) * np.array(strides_2) * np.array(strides_3))

    print('pool_size: ', pool_size)

    model.add(layers.MaxPool3D(pool_size=pool_size))
    output_5 = model.output_shape
    print('model.output_shape 5:', output_5)

    filters_4 = 1
    strides_4 = (10, 10, 1)
    kernel_size_4 = (6, 9, 3)
    model.add(layers.Conv3DTranspose(filters_4, kernel_size_4, strides=strides_4, padding='same', use_bias=False, activation='tanh'))

    output_5 = model.output_shape
    print('model.output_shape 6:', output_5)



    assert model.output_shape == (None, 480, 640, 3, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()

    kernel_size = (24, 32, 3)
    strides = (10, 10, 1)
    filters_1 = 64
    filters_2 = 128

    model.add(layers.Conv3D(filters_1, kernel_size, strides=strides, padding='same',
                                     input_shape=[480, 640, 3, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(filters_2, kernel_size, strides=strides, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = pyplot.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      pyplot.subplot(4, 4, i+1)
      pyplot.imshow(predictions[i, :, :, :, 0] * 127.5 + 127.5)
      pyplot.axis('off')

  pyplot.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  pyplot.show()


def train(dataset, epochs):

  generator = make_generator_model()
  discriminator = make_discriminator_model()
  generator_optimizer = tf.keras.optimizers.Adam(1e-4)
  discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)
  seed = tf.random.normal([num_examples_to_generate, noise_dim])



  for epoch in range(epochs):
    print('Epoch: ', epoch)
    start = time.time()

    iteractions_over_dataset = 0
    for image_batch in dataset:
      train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer)
      
      iteractions_over_dataset += 1
      print('Iteractions over dataset:', iteractions_over_dataset)
      if iteractions_over_dataset % 5 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('display images for epoch', epoch)
    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)





def main():
    dataset_dir_path, dataset_index_file_path = load_parameters()
    dataset_index = load_dataset_index(dataset_index_file_path)
    dataset = load_dataset(dataset_dir_path, dataset_index)
    train_images, train_labels = create_image_and_labels_tupla(dataset)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(len(train_images)).batch(BATCH_SIZE)
    train(train_dataset, EPOCHS)
    return dataset



if __name__ == '__main__':
    dataset = main()
    keys = [key for key in dataset]
    keys.sort()
    img = dataset[keys[0]]['img']
    print(img.shape)