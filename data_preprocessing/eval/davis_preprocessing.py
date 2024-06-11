import os
import argparse

from PIL import Image
from tqdm import tqdm


def resize_images_in_folder(folder_path, save_folder_path, H=480, W=880):
    os.makedirs(save_folder_path, exist_ok=True)

    A_save_folder = os.path.join(save_folder_path, "Annotations", "480p")
    A_target_folder = os.path.join(folder_path, "Annotations", "480p")
    os.makedirs(A_save_folder, exist_ok=True)

    for subdir in tqdm(os.listdir(A_target_folder)):
        subdir_path = os.path.join(A_target_folder, subdir)
        os.makedirs(os.path.join(A_save_folder, subdir), exist_ok=True)

        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(subdir_path, filename)
                    save_path = os.path.join(A_save_folder, subdir, filename)
                    img = Image.open(img_path)
                    resized_img = img.resize((W, H), Image.NEAREST)

                    resized_img.save(save_path)

    J_save_folder = os.path.join(save_folder_path, "JPEGImages", "480p")
    J_target_folder = os.path.join(folder_path, "JPEGImages", "480p")
    os.makedirs(J_save_folder, exist_ok=True)

    for subdir in tqdm(os.listdir(J_target_folder)):
        subdir_path = os.path.join(J_target_folder, subdir)
        os.makedirs(os.path.join(J_save_folder, subdir), exist_ok=True)

        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(subdir_path, filename)
                    save_path = os.path.join(J_save_folder, subdir, filename)

                    img = Image.open(img_path)
                    resized_img = img.resize((W, H), Image.NEAREST)
                    resized_img.save(save_path)

    os.system(f'cp -r {folder_path}/ImageSets {save_folder_path}')


parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, default='/data')

args = parser.parse_args()
resize_images_in_folder(os.path.join(args.data_root, 'DAVIS'),
                        os.path.join(args.data_root, 'DAVIS_480_880'))

