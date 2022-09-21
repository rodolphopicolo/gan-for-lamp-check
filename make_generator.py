import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import numpy as np



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


BATCH_SIZE = 2
noise_dim = 100

noise = tf.random.normal([BATCH_SIZE, noise_dim])

generator =  make_generator_model()
with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
  generated_images = generator(noise, training=True)
