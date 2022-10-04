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
import re

#import cv2 as cv

from IPython import display

from dataset_loader import load_lamps_dataset

from images_helper import generate_and_save_images
# %%



TOTAL_IMAGES_TO_LOAD = 1
BUFFER_SIZE = TOTAL_IMAGES_TO_LOAD
BATCH_SIZE = TOTAL_IMAGES_TO_LOAD
NOISE_DIM = 100
EPOCHS = 5000
num_examples_to_generate = 1

# %%
#BASE_MODEL_DIR = './models'
BASE_MODEL_DIR = '/media/rodolpho/rodolpho/models'
MODEL_DIR_PATTERN = '^model_\d{4}_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}$'

def define_model_dir(base_model_dir):
  model_dir_validation_re = re.compile(MODEL_DIR_PATTERN)

  if not os.path.exists(base_model_dir):
    os.makedirs(base_model_dir)

  model_dirs = os.listdir(base_model_dir)
  
  model_dirs = list(filter((lambda file: os.path.isdir), model_dirs))
  print(model_dirs)

  
  lambda_function = (lambda model_dir: model_dir_validation_re.match(model_dir) != None)
  dirs_filter = filter(lambda_function, model_dirs)
  model_dirs = list(dirs_filter)


  last_index = -1
  if len(model_dirs) > 0:
    model_dirs.sort()
    last_model_dir = model_dirs[-1]
    last_index = int(last_model_dir[6:10])
  model_index = str((last_index +1))
  while len(model_index) < 4:
    model_index = '0' + model_index


  now = time.localtime()
  year = str(now.tm_year)
  month = str(now.tm_mon)
  day = str(now.tm_mday)
  hour = str(now.tm_hour)
  minute = str(now.tm_min)
  second = str(now.tm_sec)
  time_zone = str(now.tm_zone)

  if len(month) < 2:
    month = '0' + month

  if len(day) < 2:
    day = '0' + day
  
  if len(hour) < 2:
    hour = '0' + hour

  if len(minute) < 2:
    minute = '0' + minute

  if len(second) < 2:
    second = '0' + second

  model_dir = 'model_' + model_index + '_' + year + '-' + month + '-' + day + 'T' + hour + ':' + minute + ':' + second
  model_dir_path = os.path.join(base_model_dir, model_dir)
  os.makedirs(model_dir_path)
  return model_dir_path


model_dir = define_model_dir(BASE_MODEL_DIR)
image_dir = os.path.join(model_dir, 'generated_images')
checkpoint_dir = os.path.join(model_dir, 'training_checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
os.makedirs(image_dir)

generator_file_path = os.path.join(model_dir, 'generator.tf')
discriminator_file_path = os.path.join(model_dir, 'discriminator.tf')

generator_summary_path = os.path.join(model_dir, 'generator_summary.txt')
discriminator_summary_path = os.path.join(model_dir, 'discriminator_summary.txt')


print('Model dir: ', model_dir, '\nImage dir: ', image_dir, '\nTraining checkpoints dir: ', checkpoint_dir, '\nCheckpoint prefix:', checkpoint_prefix)
# %%

def load_dataset(image_dir):
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
    image_name = image_dir + '/train_image_{:04d}.png'.format(i)
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

  generator.save(generator_file_path, overwrite=True, include_optimizer=True, save_format='tf')
  discriminator.save(discriminator_file_path, overwrite=True, include_optimizer=True, save_format='tf')

  with open(generator_summary_path,'w') as fh:
    generator.summary(print_fn=lambda x: fh.write(x + '\n'))

  with open(discriminator_summary_path,'w') as fh:
    discriminator.summary(print_fn=lambda x: fh.write(x + '\n'))





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
def train(dataset, epochs, cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer, image_dir, restore_last_checkpoint=False, generate_image=True):
  checkpoint = create_restore_checkpoint(restore_last_if_exists=restore_last_checkpoint)
  seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])

  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch, cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer)

    if generate_image == True:
      generate_and_save_images(generator, epoch + 1, seed, image_dir)

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

    filters = 1
    kernel_size = (16, 16, 1)
    strides = (5, 5, 1)
    model.add(layers.Conv3DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=False, activation='tanh'))

    print('Gen output shape 10:', model.output_shape)
    
    assert model.output_shape == (None, 480, 640, 3, 1)

    print('Model created.')

    return model

generator_test = make_generator_model()
noise_test = tf.random.normal([BATCH_SIZE, NOISE_DIM])
generate_and_save_images(generator_test, 0, noise_test, image_dir)

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
train_dataset = load_dataset(image_dir)

# %%
cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer = create_models()

# %%
EPOCHS = 5000
train(train_dataset, EPOCHS, cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer, image_dir, restore_last_checkpoint=True, generate_image=True)