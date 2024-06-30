import os
import shutil

def copy_images(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    for filename in os.listdir(src_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            src_file = os.path.join(src_folder, filename)
            dst_file = os.path.join(dst_folder, filename)
            shutil.copyfile(src_file, dst_file)
            print(f'Copied {src_file} to {dst_file}')

src_folder = '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Training_dataset/images'
dst_folder = '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Training_dataset_divided_expand_combined/images'

copy_images(src_folder, dst_folder)
