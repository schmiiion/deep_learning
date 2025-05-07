import os.path
import json
import random
from skimage.transform import resize
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from matplotlib.style.core import available


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
        self.available_paths = self.image_paths.copy()
        self.labels = self.get_label_dict()


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        epoch_end = False
        img_list = []
        label_list = []

        for _ in range(self.batch_size):
            if len(self.available_paths) == 0:
                image_path = random.choice(self.image_paths)
                epoch_end = True
            else:
                image_path = random.choice(self.available_paths)
                self.available_paths.remove(image_path)

            img_arr = self.load_npy_file(image_path)
            img_arr = resize(img_arr, self.image_size)
            img_list.append(img_arr)

            label_lookup = self.extract_integer(image_path)
            label_list.append(self.labels[str(label_lookup)])

        if epoch_end:
            self.epoch += 1

        return np.stack(img_list, axis=0), np.stack(label_list, axis=0)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image

        if self.rotation:
            if random.random() < 0.5:
                k = random.randint(-4, 4)
                img = np.rot90(img, k)

        if self.mirroring:
            if random.random() < 0.5:
                img = np.flip(img, axis=1)
            if random.random() < 0.5:
                img = np.flip(img, axis=0)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        print('------ class name invocation ------')
        print('class_name', x)
        print(type(x))
        return

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        img, label = self.next()
        #HERE
        pass

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
    def extract_integer(file_path):
        filename = file_path.split('/')[-1]  # Get the filename without the path
        name = filename.split('.')[0]  # Split the filename from its extension
        return int(name)  # Convert the name to an integer
