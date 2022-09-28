# -*- coding: utf-8 -*-
# https://www.tensorflow.org/tutorials/generative/dcgan

# %%
import tensorflow as tf
#import imageio
#import glob
#import glog
import numpy as np
import os
#import PIL
from tensorflow.keras import layers
import time
import matplotlib.pyplot as plt


#import cv2 as cv

from IPython import display

from dataset_loader import load_lamps_dataset

# %%
from images_helper import generate_and_save_images
# %%

TOTAL_IMAGES_TO_LOAD = 1
BUFFER_SIZE = TOTAL_IMAGES_TO_LOAD
BATCH_SIZE = TOTAL_IMAGES_TO_LOAD
NOISE_DIM = 100
EPOCHS = 5000
num_examples_to_generate = 1
checkpoint_dir = './training_checkpoints/lamps_480x640_training_checkpoints_v2'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
IMAGE_DIR = './images/lamps_480x640'

# %%

def load_dataset():
  images, dataset_metadata = load_lamps_dataset('../fotos_28_07', '../lamp-index.csv')
  train_images = images[:TOTAL_IMAGES_TO_LOAD]

  '''
  for i in range(0, len(train_images)):
    plt.axis("off")
    img = ((train_images[i] * 127.5) + 127.5).astype(dtype=int)
    plt.imshow(img)
    plt.show(block=False)

  plt.show()
  '''

  for i in range(train_images.shape[0]):
    image_name = IMAGE_DIR + '/train_image_{:04d}.png'.format(i)
    image = ((train_images[i] * 127.5) + 127.5).astype(dtype=np.uint8)
    #image = image.astype(dtype=np.uint8)
    plt.imsave(image_name, image)

  train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  return train_dataset

# %%
def discriminator_loss(real_output, fake_output, cross_entropy):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output, cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def create_restore_checkpoint(restore_last_if_exists=False):
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)

  if restore_last_if_exists == True:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

  return checkpoint;


def create_models():
  generator = make_generator_model()
  discriminator = make_discriminator_model()
  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  generator_optimizer = tf.keras.optimizers.Adam(1e-4)
  discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
  return cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer


# %%
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output, cross_entropy)
      disc_loss = discriminator_loss(real_output, fake_output, cross_entropy)

      

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# %%
def train(dataset, epochs, cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer, restore_last_checkpoint=False, generate_image=True):
  checkpoint = create_restore_checkpoint(restore_last_if_exists=restore_last_checkpoint)
  seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])

  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch, cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer)

    if generate_image == True:
      generate_and_save_images(generator, epoch + 1, seed, IMAGE_DIR)

    if (epoch + 1) % 50 == 0:
      display.clear_output(wait=False)

    if (epoch + 1) % 200 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


# %%
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(96*128*3*64, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    print('Gen output shape 3:', model.output_shape)

    model.add(layers.Reshape((96, 128, 3, 64)))

    print('Gen output shape 4:', model.output_shape)
    assert model.output_shape == (None, 96, 128, 3, 64)  # Note: None is the batch size

    filters = 32
    kernel_size = (5, 5, 3)
    strides = (1, 1, 1)
    model.add(layers.Conv3DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=False))

    print('Gen output shape 7:', model.output_shape)
    assert model.output_shape == (None, 96, 128, 3, 32)

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    

    model.add(layers.Conv3DTranspose(1, (12, 16, 3), strides=(5, 5, 1), padding='same', use_bias=False, activation='tanh'))

    print('Gen output shape 10:', model.output_shape)
    
    assert model.output_shape == (None, 480, 640, 3, 1)

    print('Model created.')

    return model

generator_test = make_generator_model()
noise_test = tf.random.normal([BATCH_SIZE, NOISE_DIM])
generate_and_save_images(generator_test, 0, noise_test, IMAGE_DIR)

# %%
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv3D(64, (48, 64, 1), strides=(12, 16, 1), padding='same', input_shape=[480, 640, 3, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(128, (48, 64, 1), strides=(12, 16, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model



# %%
train_dataset = load_dataset()

# %%
cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer = create_models()

# %%
EPOCHS = 5000
train(train_dataset, EPOCHS, cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer, restore_last_checkpoint=False, generate_image=True)
