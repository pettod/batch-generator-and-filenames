import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator as kerasImageGenerator
import math
import numpy as np
import os
from PIL import Image
import random


class ImageDataGenerator:
    def __init__(self):
        self.__train_directory = None
        self.__gt_directory = None
        self.__number_of_images_per_batch = None
        self.__number_of_patches_per_image = None
        self.__patch_size = None

    def __loadCorrespondingGtImages(self, train_batch_paths):
        gt_batch = []
        for train_image_path in train_batch_paths:
            ground_truth_image_path = "{}{}".format(
                self.__gt_directory,
                train_image_path.split(self.__train_directory)[-1])
            gt_batch.append(np.array(
                Image.open(ground_truth_image_path), dtype=np.float32))
        return np.array(gt_batch)

    def __pickRandomTrainGtPatches(self, train_batch, gt_batch):
        # Define max coordinates for image cropping
        max_x_coordinate = train_batch.shape[2] - self.__patch_size
        max_y_coordinate = train_batch.shape[1] - self.__patch_size

        # Take random patches from images
        train_patches = []
        gt_patches = []
        for i in range(train_batch.shape[0]):
            for j in range(self.__number_of_patches_per_image):
                x_start = random.randint(0, max_x_coordinate)
                x_end = x_start + self.__patch_size
                y_start = random.randint(0, max_y_coordinate)
                y_end = y_start + self.__patch_size
                train_patches.append(
                    train_batch[i, y_start:y_end, x_start:x_end])
                gt_patches.append(gt_batch[i, y_start:y_end, x_start:x_end])

        return np.array(train_patches), np.array(gt_patches)

    def batchGeneratorAndPaths(
            self, data_directory, batch_size=16, image_size=None):
        """
        Get image generator and filenames for the images in the batch

        Arguments
        --------
        data_directory : str
            Path to the data directory, directory must include folders and
            images inside those folders
        batch_size : int
            Number of images in a batch
        image_size : tuple, None
            Target image size of what will be the resized images in a batch,
            for example (256, 256). If image_size=None, the images will be
            original sized as in the folders.

        Return
        -----
        Yield tuple of numpy array of batch images and list of batch image file
        name paths
        """
        # Define image size to be original image size if not given resize
        # dimensions
        if image_size is None:
            for root, dirs, files in os.walk(data_directory, topdown=False):
                image = Image.open(os.path.join(root, files[0]))
                image_size = (image.height, image.width)
                break

        # Define generator
        datagen = kerasImageGenerator().flow_from_directory(
            data_directory, target_size=image_size, class_mode=None,
            shuffle=True, batch_size=batch_size)
        batches_per_epoch = (
            datagen.samples // datagen.batch_size +
            (datagen.samples % datagen.batch_size > 0))

        # Loop one epoch
        for i in range(batches_per_epoch):
            batch_images = next(datagen)

            # Get batch image file paths
            batch_index = ((datagen.batch_index-1) * datagen.batch_size)
            if batch_index < 0:
                if datagen.samples % datagen.batch_size > 0:
                    batch_index = max(
                        0, datagen.samples - datagen.samples %
                        datagen.batch_size)
                else:
                    batch_index = max(0, datagen.samples - datagen.batch_size)
            index_array = datagen.index_array[
                batch_index:batch_index + datagen.batch_size].tolist()
            if tf.__version__[0] == '1':
                batch_image_paths = [
                    datagen.filenames[idx] for idx in index_array]
            elif tf.__version__[0] == '2':
                batch_image_paths = [
                    datagen.filepaths[idx] for idx in index_array]

            yield batch_images, batch_image_paths

    def trainAndGtBatchGenerator(
            self, train_directory, ground_truth_directory, batch_size,
            number_of_patches_per_image=0, patch_size=None,
            normalize=False):
        """
        Take random images to batch and return (train, ground_truth) generator
        pair. If defined, take only patches from images. Train and ground truth
        file names must have same names.

        Arguments
        ---------
        train_directory: str
            Path to the train data
        ground_truth_directory : str
            Path to the ground truth data
        batch_size : int
            Number of samples in a batch
        number_of_patches_per_image : int
            If > 0 and patch_size defined, crop random patches from images,
            must be less than batch_size
        patch_size : int of None
            If defined and number_of_patches_per_image > 0, crop patches
            from images
        normalize : bool
            If true, normalize train and ground truth arrays to range [-1, 1]

        Return
        ------
        Generator of (train_images, ground_truth_images) pair
        """
        # Set member variables
        self.__train_directory = train_directory
        self.__gt_directory = ground_truth_directory
        self.__number_of_images_per_batch = batch_size
        self.__number_of_patches_per_image = number_of_patches_per_image
        self.__patch_size = patch_size

        # Take either images in batch or patches in batch
        if number_of_patches_per_image > 0 and patch_size is not None:
            self.__number_of_images_per_batch = math.ceil(
                batch_size / self.__number_of_patches_per_image)
        batch_generator = self.batchGeneratorAndPaths(
            self.__train_directory, self.__number_of_images_per_batch)

        # Take respective ground truth images
        while True:
            train_batch, train_image_paths = next(batch_generator)
            gt_batch = self.__loadCorrespondingGtImages(train_image_paths)
            if self.__patch_size is not None and \
                    self.__number_of_patches_per_image > 0:
                train_batch, gt_batch = self.__pickRandomTrainGtPatches(
                    train_batch, gt_batch)
            if normalize:
                train_batch = self.normalizeArray(train_batch)
                gt_batch = self.normalizeArray(gt_batch)
            yield train_batch, gt_batch

    def numberOfBatchesPerEpoch(
            self, train_directory, batch_size,
            number_of_patches_per_image=0):
        number_of_images_per_batch = batch_size
        if number_of_patches_per_image > 0:
            number_of_images_per_batch = math.ceil(
                batch_size / number_of_patches_per_image)

        # Define generator same way as the training batch generator
        datagen = kerasImageGenerator().flow_from_directory(
            train_directory, target_size=(256, 256), class_mode=None,
            shuffle=True, batch_size=number_of_images_per_batch)
        batches_per_epoch = (
            datagen.samples // datagen.batch_size +
            (datagen.samples % datagen.batch_size > 0))
        return batches_per_epoch

    def normalizeArray(self, data_array, max_value=255):
        return (data_array / max_value - 0.5) * 2

    def unnormalizeArray(self, data_array, max_value=255):
        data_array = (data_array / 2 + 0.5) * max_value
        data_array[data_array < 0.0] = 0.0
        data_array[data_array > max_value] = max_value
        return data_array
