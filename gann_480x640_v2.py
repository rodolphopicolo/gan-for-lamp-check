# -*- coding: utf-8 -*-
# https://www.tensorflow.org/tutorials/generative/dcgan

# %%
import os
import sys
import time
import re

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

from IPython import display

from dataset_loader import load_lamps_dataset
from images_helper import generate_and_save_images, check_previous_epochs



QUANTITY_OF_EPOCHS_TO_SAVE_CHECKPOINT = 500
TOTAL_IMAGES_TO_LOAD = 1
BUFFER_SIZE = TOTAL_IMAGES_TO_LOAD
BATCH_SIZE = TOTAL_IMAGES_TO_LOAD
NOISE_DIM = 100
num_examples_to_generate = 1
IMAGE_NAME_PREFIX = 'image_at_epoch_'
NOISE_IMAGE_PREFIX = 'generator_test_'

MODEL_DIR = 'model_dir'
IMAGE_DIR = 'image_dir'
CHECKPOINT_DIR = 'checkpoint_dir'
CHECKPOINT_PREFIX = 'checkpoint_prefix'
GENERATOR_FILE_PATH = 'generator_file_path'
DISCRIMINATOR_FILE_PATH = 'discriminator_file_path'
GENERATOR_SUMMARY_PATH = 'generator_summary_path'
DISCRIMINATOR_SUMMARY_PATH = 'discriminator_summary_path'
MODEL_VERSION_FILE_PATH = 'model_version_file_path'

EPOCH_STEP_TO_GENERATE_IMAGE = 10


#BASE_MODEL_DIR = './models'
BASE_MODEL_DIR = '/home/rodolpho/Documents/mest/GAN/application/app/models'
MODEL_DIR_PATTERN = '^model_\d{4}_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}$'

def define_model_dir(base_model_dir):
  model_dir_validation_re = re.compile(MODEL_DIR_PATTERN)

  os.makedirs(base_model_dir, exist_ok=True)

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
  os.makedirs(model_dir_path, exist_ok=True)
  return model_dir_path



def define_paths(previous_generated_model_dir=None):
  if previous_generated_model_dir == None:
    model_dir = define_model_dir(BASE_MODEL_DIR)
  else:
    model_dir = previous_generated_model_dir

  image_dir = os.path.join(model_dir, 'generated_images')
  checkpoint_dir = os.path.join(model_dir, 'training_checkpoints')
  checkpoint_prefix = 'ckpt'
  os.makedirs(image_dir, exist_ok=True)

  generator_file_path = os.path.join(model_dir, 'generator.tf')
  discriminator_file_path = os.path.join(model_dir, 'discriminator.tf')

  generator_summary_path = os.path.join(model_dir, 'generator_summary.txt')
  discriminator_summary_path = os.path.join(model_dir, 'discriminator_summary.txt')

  model_version_file_path = os.path.join(model_dir, 'models_version.txt')

  paths = {
    MODEL_DIR: model_dir,
    IMAGE_DIR: image_dir,
    CHECKPOINT_DIR: checkpoint_dir,
    CHECKPOINT_PREFIX: checkpoint_prefix,
    GENERATOR_FILE_PATH: generator_file_path,
    DISCRIMINATOR_FILE_PATH: discriminator_file_path,
    GENERATOR_SUMMARY_PATH: generator_summary_path,
    DISCRIMINATOR_SUMMARY_PATH: discriminator_summary_path,
    MODEL_VERSION_FILE_PATH: model_version_file_path,
  }

  return paths


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


def discriminator_loss(real_output, fake_output, cross_entropy):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output, cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def create_restore_checkpoint_manager(checkpoint_dir, checkpoint_prefix, generator, generator_optimizer, discriminator, discriminator_optimizer, max_checkpoint_to_keep=2, restore_last_if_exists=True):
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)

  manager = tf.train.CheckpointManager(checkpoint
                                      , directory=checkpoint_dir
                                      , checkpoint_name=checkpoint_prefix
                                      , max_to_keep=max_checkpoint_to_keep)

  if restore_last_if_exists == True:
    checkpoint.restore(manager.latest_checkpoint)

  return manager;



def check_model_version_file_line(line, model_version, valid_model_versions, model_type):

  if model_version == None:
    raise Exception('Model version not specified')

  previous_version = None
  line = line.upper().strip()
  if len(line) > len(model_type):
    if line[0: len(model_type)] == model_type.upper():
      splitted = line.split(':')
      if len(splitted) != 2:
        raise Exception('Invalid model version file: invalid ' + model_type.lower() + ' version, colon (":") appears more than once')
      right_side = splitted[1].strip()
      if right_side.isnumeric() == False:
        raise Exception('Invalid model version file: invalid ' + model_type.lower() + ' version, version is not a number')
      previous_version = int(right_side)
      if previous_version not in valid_model_versions:
        raise Exception('Invalid model version file: version ' + str(previous_version) + ' is not valid, valid values are ' + str(valid_model_versions))

  if previous_version != None:
    if previous_version != model_version:
      raise Exception('Invalid model version: the right version for ' + model_type.lower() + ' is ' + str(previous_version))

  return previous_version


def check_models_version(paths, generator_version, discriminator_version):

  if generator_version == None:
    raise Exception('Generator version not specified')

  if discriminator_version == None:
    raise Exception('Discriminator version not specified')

  if generator_version not in GENERATOR_VERSIONS:
    raise Exception('Invalid generator_model version ', generator_version)

  if discriminator_version not in DISCRIMINATOR_VERSIONS:
    raise Exception('Invalid discriminator_model version ', discriminator_version)

  model_version_file_path = paths[MODEL_VERSION_FILE_PATH]
  if os.path.exists(model_version_file_path) == False:
    return

  with open(model_version_file_path,'r') as f:
    lines = f.readlines()

  previous_generator_version = None
  previous_discriminator_version = None

  for line in lines:
    line_version = check_model_version_file_line(line, generator_version, GENERATOR_VERSIONS, 'generator')
    if line_version != None:
      if previous_generator_version != None:
        raise Exception('Invalid model version file: generator version appears more than once')
      else:
        previous_generator_version = line_version

    line_version = check_model_version_file_line(line, discriminator_version, DISCRIMINATOR_VERSIONS, 'discriminator')
    if line_version != None:
      if previous_discriminator_version != None:
        raise Exception('Invalid model version file: discriminator version appears more than once')
      else:
        previous_discriminator_version = line_version



def save_model_version(paths, generator_version, discriminator_version):
  model_version_file_path = paths[MODEL_VERSION_FILE_PATH]

  if os.path.exists(model_version_file_path):
    check_models_version(paths, generator_version, discriminator_version)
    return

  with open(model_version_file_path,'w') as fh:
    fh.write('Generator version: ' + str(generator_version))
    fh.write('\nDiscriminator version: ' + str(discriminator_version))


def create_models(paths, generator_version, discriminator_version):

  save_model_version(paths, generator_version, discriminator_version)

  generator = make_generator_model(generator_version)
  discriminator = make_discriminator_model(discriminator_version)
  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  generator_optimizer = tf.keras.optimizers.Adam(1e-4)
  discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

  if os.path.exists(paths[GENERATOR_FILE_PATH])==False:
    generator.save(paths[GENERATOR_FILE_PATH], overwrite=False, include_optimizer=True, save_format='tf')
  if os.path.exists(paths[DISCRIMINATOR_FILE_PATH])==False:
    discriminator.save(paths[DISCRIMINATOR_FILE_PATH], overwrite=False, include_optimizer=True, save_format='tf')

  if os.path.exists(paths[GENERATOR_SUMMARY_PATH]) == False:
    with open(paths[GENERATOR_SUMMARY_PATH],'w') as fh:
      generator.summary(print_fn=lambda x: fh.write(x + '\n'))

  if os.path.exists(paths[DISCRIMINATOR_SUMMARY_PATH]) == False:
    with open(paths[DISCRIMINATOR_SUMMARY_PATH],'w') as fh:
      discriminator.summary(print_fn=lambda x: fh.write(x + '\n'))

  noise_test = tf.random.normal([BATCH_SIZE, NOISE_DIM])
  generate_and_save_images(generator, 0, noise_test, paths[IMAGE_DIR], NOISE_IMAGE_PREFIX)

  return cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer



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


def train(dataset, epochs, cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer, paths, restore_last_checkpoint=True, generate_image=True, previous_calculated_epoch=0, max_checkpoint_to_keep=2):

  checkpoint_dir = paths[CHECKPOINT_DIR]
  checkpoint_prefix = paths[CHECKPOINT_PREFIX]
  image_dir = paths[IMAGE_DIR]


  checkpoint_manager = create_restore_checkpoint_manager(checkpoint_dir, checkpoint_prefix, generator, generator_optimizer, discriminator, discriminator_optimizer, max_checkpoint_to_keep, restore_last_if_exists=restore_last_checkpoint)
  seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])

  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch, cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer)

    if generate_image == True and ((epoch + 1 + previous_calculated_epoch) % EPOCH_STEP_TO_GENERATE_IMAGE == 0):
      generate_and_save_images(generator, epoch + 1 + previous_calculated_epoch, seed, image_dir, IMAGE_NAME_PREFIX)

    if (epoch + 1) % 50 == 0:
      display.clear_output(wait=False)

    if (epoch + 1) % QUANTITY_OF_EPOCHS_TO_SAVE_CHECKPOINT == 0:
      checkpoint_manager.save()

    print ('Time for epoch {} is {} sec'.format(epoch + 1 + previous_calculated_epoch, time.time()-start))


def make_generator_model(version):
  if version == 1:
    return generator_model_v1()
  elif version == 2:
    return generator_model_v2()
  elif version == 3:
    return generator_model_v3()

GENERATOR_VERSIONS = {1, 2, 3}

def generator_model_v1():
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
    
    filters_2 = 1
    kernel_size_2 = (12, 16, 3)
    strides_2 = (5, 5, 1)
    model.add(layers.Conv3DTranspose(filters_2, kernel_size_2, strides=strides_2, padding='same', use_bias=False, activation='tanh'))

    print('Gen output shape 10:', model.output_shape)
    
    assert model.output_shape == (None, 480, 640, 3, 1)

    print('Model created.')

    return model

def generator_model_v2():
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

def generator_model_v3():
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
    strides = (10, 10, 1)
    model.add(layers.Conv3DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=False, activation='tanh'))

    model.add(layers.MaxPooling3D(pool_size=(2, 2, 1)))

    print('Gen output shape 10:', model.output_shape)
    
    assert model.output_shape == (None, 480, 640, 3, 1)

    print('Model created.')

    return model


def make_discriminator_model(discriminator_version):
  if discriminator_version == 1:
    return discriminator_model_v1()
  elif discriminator_version == 2:
    return discriminator_model_v2()
  elif discriminator_version == 3:
    return discriminator_model_v3()

DISCRIMINATOR_VERSIONS = {1, 2, 3}

def discriminator_model_v1():
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

def discriminator_model_v2():
    model = tf.keras.Sequential()
    model.add(layers.Conv3D(64, (48, 48, 1), strides=(8, 8, 3), padding='same', input_shape=[480, 640, 3, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(128, (48, 48, 1), strides=(8, 8, 3), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def discriminator_model_v3():
    model = tf.keras.Sequential()
    model.add(layers.Conv3D(64, (12, 12, 1), strides=(4, 4, 1), padding='same', input_shape=[480, 640, 3, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(128, (12, 12, 1), strides=(4, 4, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def run(generator_version=1, discriminator_version=1, epochs = 100000, max_checkpoint_to_keep=2, previous_generated_model_dir = None):
  paths = define_paths(previous_generated_model_dir)

  check_models_version(paths, generator_version, discriminator_version)

  previous_calculated_epoch = 0
  if previous_generated_model_dir != None:
    previous_calculated_epoch = check_previous_epochs(paths[IMAGE_DIR], 'image_at_epoch_')

  

  train_dataset = load_dataset(paths[IMAGE_DIR])
  cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer = create_models(paths, generator_version, discriminator_version)
  train(train_dataset, epochs, cross_entropy, generator, generator_optimizer, discriminator, discriminator_optimizer, paths, restore_last_checkpoint=True, generate_image=True, previous_calculated_epoch=previous_calculated_epoch, max_checkpoint_to_keep=max_checkpoint_to_keep)

def run_config_1():
  epochs=10000
  generator_version=2
  discriminator_version=1
  max_checkpoint_to_keep=1
  previous_generated_model_dir = '/home/rodolpho/Documents/mest/GAN/application/app/models/model_0000_2022-10-04T14:55:55'
  run(generator_version=generator_version, discriminator_version=discriminator_version, epochs=epochs, max_checkpoint_to_keep=max_checkpoint_to_keep, previous_generated_model_dir=previous_generated_model_dir)

def run_config_2():
  epochs=10000
  generator_version=2
  discriminator_version=3
  max_checkpoint_to_keep=1
  previous_generated_model_dir = '/home/rodolpho/Documents/mest/GAN/application/app/models/model_0000_2022-10-08T07:43:29'
  run(generator_version=generator_version, discriminator_version=discriminator_version, epochs=epochs, max_checkpoint_to_keep=max_checkpoint_to_keep, previous_generated_model_dir=previous_generated_model_dir)
# %%

def check_config():
  arg_id = '--config='
  for i in range(len(sys.argv)):
    arg = sys.argv[i]
    if len(arg) > len(arg_id):
      if arg[:len(arg_id)] == arg_id:
        config = arg[len(arg_id):]
        if config.isnumeric == False:
          print('Invalid config: specify a number between 1 and 2')
          return -2

        config = int(config)

        if config < 1 or config > 2:
          print('Invalid config: specify a number between 1 and 2')
          return -3

        print('Config: ', config)
        return config

  print('Invalid not specified')
  return -1

def main():
  config = check_config()
  if config <= 0:
    return

  if config == 1:
    print('running config 1')
    run_config_1()
  elif config ==2:
    print('running config 2')
    run_config_2()
  else:
    raise Exception('Unsupported config ' + str(config))


if __name__ == '__main__':
  main()
# %%
