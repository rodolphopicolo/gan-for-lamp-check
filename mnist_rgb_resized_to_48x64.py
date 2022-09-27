# -*- coding: utf-8 -*-
# https://www.tensorflow.org/tutorials/generative/dcgan

# %%

import tensorflow as tf

import glob
import glog
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

import cv2 as cv


from IPython import display

from dataset_loader import load_lamps_dataset, load_mnist_dataset
# %%
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_labels = None # to constat it is not important

train_images = train_images[:2]

# %%

train_images_48x64 = np.zeros((train_images.shape[0], 48, 64, 3), dtype=float)
train_images_48x64[:, :, :, :] = [100, 255, 100]
train_images_48x64[:, :28, :28, :] = [0, 0, 0]

train_images_48x64[:, :train_images.shape[1], :train_images.shape[2], 0] = train_images

train_images_48x64 = (train_images_48x64 - 127.5) / 127.5

train_images = np.zeros((train_images.shape), dtype=float)

train_images = train_images_48x64


# %%

for i in range(0, len(train_images)):
  plt.axis("off")
  img = ((train_images[i] * 127.5) + 127.5).astype(dtype=int)
  plt.imshow(img)
  plt.show()


# %%
#BUFFER_SIZE = 60000
BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE = 1

#train_images = train_images[0:5]

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# %%

NOISE_DIM = 512

# %%

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(12*16*3*256, use_bias=False, input_shape=(NOISE_DIM,)))

    print('Gen output shape 1:', model.output_shape)

    model.add(layers.BatchNormalization())

    print('Gen output shape 2:', model.output_shape)

    model.add(layers.LeakyReLU())

    print('Gen output shape 3:', model.output_shape)

    model.add(layers.Reshape((12, 16, 3, 256)))

    print('Gen output shape 4:', model.output_shape)

    assert model.output_shape == (None, 12, 16, 3, 256)  # Note: None is the batch size

    filters = 128
    kernel_size = (5, 5, 5)
    strides = (1, 1, 1)
    model.add(layers.Conv3DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=False))

    print('Gen output shape 5:', model.output_shape)

    assert model.output_shape == (None, 12, 16, 3, 128)

    model.add(layers.BatchNormalization())

    print('Gen output shape 6:', model.output_shape)

    model.add(layers.LeakyReLU())

    print('Gen output shape 7:', model.output_shape)

    model.add(layers.Conv3DTranspose(64, (5, 5, 5), strides=(2, 2, 2), padding='same', use_bias=False))

    print('Gen output shape 8:', model.output_shape)

    assert model.output_shape == (None, 24, 32, 6, 64)

    model.add(layers.BatchNormalization())

    print('Gen output shape 9:', model.output_shape)

    model.add(layers.LeakyReLU())

    print('Gen output shape 10:', model.output_shape)

    model.add(layers.Conv3DTranspose(1, (5, 5, 5), strides=(2, 2, 2), padding='same', use_bias=False, activation='tanh'))

    print('Gen output shape 11:', model.output_shape)

    model.add(layers.MaxPooling3D(pool_size=(1, 1, 4)))

    print('Gen output shape 12:', model.output_shape)

    assert model.output_shape == (None, 48, 64, 3, 1)

    print('Model created.')

    return model


# %%
def convert_generated_images_to_visible_mode(generated_images):
    images = generated_images.numpy()
    min_value = images.min()
    max_value = images.max()
    diff = max_value - min_value
    proportion = 255 / diff
    images = images + -min_value
    images = images * proportion
    images = images.astype(dtype=int)
    return images[:, :, :, :, 0]


# %%

generator = make_generator_model()

noise = tf.random.normal([1, NOISE_DIM])


generated_image = generator(noise, training=False)

images = convert_generated_images_to_visible_mode(generated_image)
plt.axis("off")
plt.imshow(images[0])
plt.show()




# %%

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv3D(64, (5, 5, 3), strides=(2, 2, 2), padding='same', input_shape=[48, 64, 3, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(128, (5, 5, 3), strides=(2, 2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# %%

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

# %%


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# %%

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# %%

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# %%

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# %%

checkpoint_dir = './training_checkpoints/mnist_rgb_resized_48x64_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# %%

EPOCHS = 5000
num_examples_to_generate = 1

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])

# %%

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

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

# %%

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=False)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 50 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=False)
  generate_and_save_images(generator,
                           epochs,
                           seed)

# %%


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  #fig = plt.figure(figsize=(4, 4))

  
  images = convert_generated_images_to_visible_mode(predictions)
  for i in range(predictions.shape[0]):
    image_name = './images/mnist_rgb_resized_48x64_images/image_at_epoch_{:04d}.png'.format(epoch)
    image = images[i]
    plt.axis("off")
    plt.imshow(image)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,  hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(image_name, bbox_inches='tight', pad_inches = 0)
    plt.show()


# %%

train(train_dataset, EPOCHS)

# %%

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


display_image(EPOCHS)

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)


import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)



'''
def main():
    checkpoint, checkpoint_prefix = prepare_checkpoint()
    train_dataset = load_mnist_dataset()
    train(train_dataset, EPOCHS, checkpoint, checkpoint_prefix)
    return train_dataset

if __name__ == '__main__':
    dataset = main()
    print(dataset.shape)
'''
