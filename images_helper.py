import os
import re
import numpy as np
import matplotlib.pyplot as plt
import shutil
import imageio

def calculate_next_image_name(epoch, prediction, image_dir, image_name_prefix, overwrite=False):
  execution = 0
  while True:
    image_name = image_dir + '/' + image_name_prefix + '{:06d}_pred_{:04d}_exec_{:04d}.png'.format(epoch, prediction, execution)
    if overwrite == True:
      return image_name
    elif os.path.exists(image_name) == False:
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
    images = images[:, :, :, :, 0]

    images_rgb = np.zeros(images.shape, dtype=np.uint8)

    images_rgb[:,:,:,0] = images[:,:,:,2]
    images_rgb[:,:,:,1] = images[:,:,:,1]
    images_rgb[:,:,:,2] = images[:,:,:,0]


    return images_rgb


def generate_and_save_images(model, epoch, seed, image_dir, image_name_prefix):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(seed, training=False)

  #fig = plt.figure(figsize=(4, 4))

  
  images = convert_generated_images_to_visible_mode(predictions)
  for i in range(predictions.shape[0]):
    
    image_name = calculate_next_image_name(epoch, i, image_dir, image_name_prefix)

    image = images[i]
    #image = image.astype(dtype=np.uint8)
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
  image_name_pattern = '^' + image_name_prefix + '\d{4,6}_pred_\d{4}_exec_\d{4}.png$'

  image_name_validation_re = re.compile(image_name_pattern)

  images = os.listdir(image_dir)

  if len(images) == 0:
    return []

  images = list(filter((lambda file: os.path.isfile), images))

  if len(images) == 0:
    return []

  lambda_function = (lambda image_name: image_name_validation_re.match(image_name) != None)
  images = list(filter(lambda_function, images))

  if len(images) == 0:
    return []

  images.sort()
  
  return images

def keep_just_images_from_epochs_multiples_of(image_dir, image_name_prefix, multiple, destination_dir):
    images = generated_images_list(image_dir, image_name_prefix)
    for image_name in images:
      epoch = epoch_from_image_name(image_name, image_name_prefix)


      if epoch % multiple != 0:
        print('REMOVE', image_name, epoch)
        source_absolute_path = os.path.join(image_dir, image_name)
        destination_absolute_path = os.path.join(destination_dir, image_name)
        shutil.move(source_absolute_path, destination_absolute_path)
      else:
        print('KEEP', image_name, epoch)

def rename_images_epoch_4_digits_to_6_digits(image_dir, image_name_prefix):
  images = generated_images_list(image_dir, image_name_prefix)
  for image_name in images:
      epoch = epoch_from_image_name(image_name, image_name_prefix)
      absolute_new_name = calculate_next_image_name(epoch, 0, image_dir, image_name_prefix, overwrite=True)
      print(image_name)
      print(absolute_new_name[-45:])

      if absolute_new_name[-len(image_name):] == image_name:
        print('Same images\n-----------------------')
        continue

      shutil.move(os.path.join(image_dir, image_name), absolute_new_name)
      print('-----------------------')


def epoch_from_image_name(image_name, image_name_prefix):
  epoch = image_name[len(image_name_prefix): len(image_name_prefix) + 6]
  while epoch[-1].isnumeric() == False:
    epoch = epoch[:-1]

  epoch = int(epoch)

  return epoch

def check_previous_epochs(image_dir, image_name_prefix):
  images = generated_images_list(image_dir, image_name_prefix)
  if len(images) == 0:
    return 0

  last_image_name = images[-1]

  last_epoch = epoch_from_image_name(last_image_name, image_name_prefix)
  last_epoch = int(last_epoch)
  print('Generated image for last epoch:', last_image_name, 'Epoch:', last_epoch)

  return last_epoch

def create_epochs_animation(image_dir, image_name_prefix, epoch_multiple_of):
  filenames = generated_images_list(image_dir, image_name_prefix)
  if len(filenames) == 0:
    return None

  anim_file = '/tmp/anime/dcgan.gif'
  filenames = sorted(filenames)

  with imageio.get_writer(anim_file, mode='I') as writer:
    counter = 0
    for filename in filenames:
      counter += 1
      if counter % 10 == 0:
        print('Processing image ', counter, ' of ', len(filenames))

      epoch = epoch_from_image_name(filename, image_name_prefix)
      if epoch % epoch_multiple_of != 0:
        continue

      image = imageio.imread(os.path.join(image_dir, filename))
      writer.append_data(image)

      

    image = imageio.imread(os.path.join(image_dir, filename))
    for i in range(20):
      writer.append_data(image)

  import tensorflow_docs.vis.embed as embed
  embed.embed_file(anim_file)




if __name__ == '__main__':

  image_dir = '/home/rodolpho/Documents/mest/GAN/application/app/models/model_0000_2022-10-04T14:55:55/generated_images'
  image_name_prefix = 'image_at_epoch_'

  #rename_images_epoch_4_digits_to_6_digits(image_dir, image_name_prefix)

  #last_epoch = check_previous_epochs('/home/rodolpho/Documents/mest/GAN/application/app/models/model_0000_2022-10-04T14:55:55/generated_images', 'image_at_epoch_')
  #print(last_epoch)

  #destination_dir = '/home/rodolpho/Documents/mest/GAN/application/app/models/model_0000_2022-10-04T14:55:55/not_multiple_images'
  #keep_just_images_from_epochs_multiples_of('/home/rodolpho/Documents/mest/GAN/application/app/models/model_0000_2022-10-04T14:55:55/generated_images', 'image_at_epoch_', 10, destination_dir)

  create_epochs_animation('/home/rodolpho/Documents/mest/GAN/application/app/models/model_0000_2022-10-04T14:55:55/generated_images', 'image_at_epoch_', 100)