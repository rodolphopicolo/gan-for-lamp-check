import os
import shutil

models_dir_path = './app/models'
output_dir_path = './generated_images'

PREDICTED_IMAGES = 'predicted_images'
CONFIG_0009_LABEL = 'config_0009_label'

print('Working dir: ' + os.getcwd())
files = os.listdir(models_dir_path)
filtered_files = [f for f in files if f[0: len(CONFIG_0009_LABEL)] == CONFIG_0009_LABEL]
filtered_files.sort()

for file in filtered_files:
    
    label = int(file[18:])
    print(file + ' - label: ' + str(label))
    predicted_images_path = os.path.join(models_dir_path, file, PREDICTED_IMAGES)
    images = os.listdir(predicted_images_path)
    images.sort()

    count = 0
    for image_current_name in images:
        count = count + 1
        image_new_name = 'lampada_' + ('0' if label < 10 else '') + str(label) + '_seq_' + ('00' if count < 10 else '0' if count < 100 else '') + str(count)

        print(image_current_name, image_new_name)

        source_file = os.path.join(predicted_images_path, image_current_name)
        destination_file = os.path.join(output_dir_path, image_new_name)

        print(source_file, destination_file)
        shutil.copyfile(source_file, destination_file)