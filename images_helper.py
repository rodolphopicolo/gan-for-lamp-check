import os
import re
import numpy as np
import matplotlib.pyplot as plt
import shutil

def calculate_next_image_name(epoch, prediction, image_dir, image_name_prefix):
  execution = 0
  while True:
    image_name = image_dir + '/' + image_name_prefix + '{:04d}_pred_{:04d}_exec_{:04d}.png'.format(epoch, prediction, execution)
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


def generate_and_save_images(model, epoch, test_input, image_dir, image_name_prefix):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  #fig = plt.figure(figsize=(4, 4))

  
  images = convert_generated_images_to_visible_mode(predictions)
  for i in range(predictions.shape[0]):
    
    image_name = calculate_next_image_name(epoch, i, image_dir, image_name_prefix)

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
    
def generated_images_list(image_dir, image_name_prefix):
  image_name_pattern = image_name_prefix + '\d{4}' + '_pred_' + '\d{4}' + '_exec_' + '\d{4}' + '.png'

  image_name_validation_re = re.compile(image_name_pattern)

  images = os.listdir(image_dir)

  if len(images) == 0:
    return []

  images = list(filter((lambda file: os.path.isfile), images))

  if len(images) == 0:
    return []

  lambda_function = (lambda model_dir: image_name_validation_re.match(model_dir) != None)
  images = list(filter(lambda_function, images))

  if len(images) == 0:
    return []

  images.sort()
  
  return images

def keep_just_images_from_epochs_multiples_of(image_dir, image_name_prefix, multiple, destination_dir):
    images = generated_images_list(image_dir, image_name_prefix)
    for image_name in images:
      epoch = int(image_name[len(image_name_prefix): len(image_name_prefix) + 4])

      if epoch % multiple != 0:
        print('REMOVE', image_name, epoch)
        source_absolute_path = os.path.join(image_dir, image_name)
        destination_absolute_path = os.path.join(destination_dir, image_name)
        shutil.move(source_absolute_path, destination_absolute_path)
      else:
        print('KEEP', image_name, epoch)

      



def check_previous_epochs(image_dir, image_name_prefix):
  images = generated_images_list(image_dir, image_name_prefix)
  if len(images) == 0:
    return 0

  last_image_name = images[-1]

  last_epoch = last_image_name[len(image_name_prefix): len(image_name_prefix) + 4]
  last_epoch = int(last_epoch)
  print('Generated image for last epoch:', last_image_name, 'Epoch:', last_epoch)

  return last_epoch


if __name__ == '__main__':
  last_epoch = check_previous_epochs('/home/rodolpho/Documents/mest/GAN/application/app/models/model_0000_2022-10-04T14:55:55/generated_images', 'image_at_epoch_')
  print(last_epoch)

  destination_dir = '/home/rodolpho/Documents/mest/GAN/application/app/models/model_0000_2022-10-04T14:55:55/not_multiple_images'
  keep_just_images_from_epochs_multiples_of('/home/rodolpho/Documents/mest/GAN/application/app/models/model_0000_2022-10-04T14:55:55/generated_images', 'image_at_epoch_', 10, destination_dir)