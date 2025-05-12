import os.path
import json
import random
from skimage.transform import resize
#import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path: str = file_path
        self.label_path: str= label_path
        self.batch_size: int = batch_size
        self.image_size: List[int] = image_size
        self.rotation: bool= rotation
        self.mirroring: bool = mirroring
        self.shuffle: bool = shuffle

        self.epoch = 0
        self.image_paths = self.get_npy_paths()
        if self.shuffle:
            random.shuffle(self.image_paths)
            self.available_paths = self.image_paths.copy()
        else:
            self.img_counter = 0


        self.labels = self.get_label_dict()


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        epoch_end = False
        img_list = []
        label_list = []

        for img_idx in range(self.batch_size):
            if self.shuffle:
                if len(self.available_paths) == 0:
                    image_path = random.choice(self.image_paths)
                    epoch_end = True
                else:
                    image_path = random.choice(self.available_paths)
                    self.available_paths.remove(image_path)

            else:
                image_path = self.image_paths[self.img_counter]
                new_counter = self.img_counter + 1
                new_counter_capped = new_counter % len(self.image_paths)
                if new_counter_capped == 0:
                    self.epoch +=1
                self.img_counter = new_counter_capped

            img_arr = self.load_npy_file(image_path)
            img_arr = resize(img_arr, self.image_size)
            img_arr = self.augment(img_arr, self.rotation, self.mirroring)
            img_list.append(img_arr)

            img_number_for_lookup = self.extract_label_from_filename(image_path)
            label_as_int = self.labels[img_number_for_lookup]
            label_list.append(label_as_int)

        if epoch_end:
            self.epoch += 1
            if self.shuffle:
                random.shuffle(self.image_paths)

        return np.stack(img_list, axis=0), np.stack(label_list, axis=0)

    def augment(self,img, rotation=False, mirroring=False):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if rotation:
            if random.random() < 0.5:
                k = random.randint(-4, 4)
                img = np.rot90(img, k)

        if mirroring:
            if random.random() < 0.5:
                img = np.flip(img, axis=1)
            if random.random() < 0.5:
                img = np.flip(img, axis=0)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, int_label):
        # This function returns the class name for a specific input
        return self.class_dict[int_label]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        imgs, labels = self.next()

        n = len(imgs)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axs = axs.ravel()  # Flatten for easy iteration

        for ax in axs:
            ax.axis('off')  # Hide axes for empty subplots

        for i, (img, label) in enumerate(zip(imgs, labels)):
            axs[i].imshow(img)
            label_str = self.class_name(label)
            axs[i].set_title(label_str, fontsize=9)

        plt.tight_layout()
        plt.show()

    def get_npy_paths(self):
        npy_paths = []
        for root, dirs, files in os.walk(self.file_path):
            for file in files:
                if file.endswith('.npy'):
                    npy_paths.append(os.path.join(root, file))
        return npy_paths

    def get_label_dict(self):
        try:
            with open(self.label_path, 'r') as file:
                data_dict = json.load(file)
                return data_dict
        except FileNotFoundError:
            print(f"File {self.label_path} not found.")
            return None
        except json.JSONDecodeError:
            print(f"Failed to parse JSON in file {self.label_path}.")
            return None

    @staticmethod
    def load_npy_file(file_path):
        try:
            loaded_array = np.load(file_path)
            return loaded_array
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return None
        except Exception as e:
            print(f"Failed to load file {file_path}: {e}")
            return None

    @staticmethod
    def extract_label_from_filename(file_path):
        filename = file_path.split('/')[-1]  # Get the filename without the path
        name = filename.split('.')[0]  # Split the filename from its extension
        return name  # Convert the name to an integer
