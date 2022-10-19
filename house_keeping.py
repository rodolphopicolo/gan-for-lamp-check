from genericpath import isdir
import os
from images_helper import keep_just_images_from_epochs_multiples_of

def remove_old_images():
  image_name_prefix = 'image_at_epoch_'
  multiple = 20
  base_dir = '/home/rodolpho/Documents/mest/GAN/application/app/images'
  destination_dir = '/home/rodolpho/Documents/mest/GAN/application/app/images/removed_images'
  

  old_image_dirs = os.listdir(base_dir)
  old_image_dirs.sort()
  #keep_just_images_from_epochs_multiples_of
  for dir in old_image_dirs:
    print(dir)
    absolute_dir_path = os.path.join(base_dir, dir)
    if os.path.isdir(absolute_dir_path) == False:
      continue
    if dir == 'removed_images':
      continue

    keep_just_images_from_epochs_multiples_of(absolute_dir_path, image_name_prefix, multiple, destination_dir)

if __name__ == '__main__':
    remove_old_images()
