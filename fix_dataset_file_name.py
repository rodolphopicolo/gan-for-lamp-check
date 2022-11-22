import os


dir_path = '../fotos_28_07'
files = os.listdir(dir_path)


for file in files:
    current_file_name = os.path.join(dir_path, file)
    photo_id = int(file[5:-4])
    new_file = 'foto_{:04d}.png'.format(photo_id)

    photo_id_with_replace = file.replace('Foto_', '').replace('.png', '')
    photo_id_with_replace = int(photo_id_with_replace)
    if photo_id != photo_id_with_replace:
        print('Error')

    new_file_name = os.path.join(dir_path, new_file)

    print(file, new_file, current_file_name, new_file_name)
    os.rename(current_file_name, new_file_name)

