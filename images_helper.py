import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_next_image_name(epoch, prediction, image_dir):
  IMAGE_NAME_PREFIX = 'image_at_epoch_'
  execution = 0
  while True:
    image_name = image_dir + '/' + IMAGE_NAME_PREFIX + '{:04d}_pred_{:04d}_exec_{:04d}.png'.format(epoch, prediction, execution)
    if not os.path.exists(image_name):
      return image_name
    execution += 1


def show_image(image):
  plt.axis("off")
  plt.imshow(image)
  plt.gca().set_axis_off()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,  hspace = 0, wspace = 0)
  plt.margins(0,0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.show(block=False)

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


def generate_and_save_images(model, epoch, test_input, image_dir):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  #fig = plt.figure(figsize=(4, 4))

  
  images = convert_generated_images_to_visible_mode(predictions)
  for i in range(predictions.shape[0]):
    
    image_name = calculate_next_image_name(epoch, i, image_dir)

    image = images[i]
    image = image.astype(dtype=np.uint8)
    #plt.axis("off")
    #plt.imshow(image)
    #plt.gca().set_axis_off()
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,  hspace = 0, wspace = 0)
    #plt.margins(0,0)
    #plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #plt.savefig(image_name, bbox_inches='tight', pad_inches = 0)
    #plt.show(block=False)
    plt.imsave(image_name, image)
    