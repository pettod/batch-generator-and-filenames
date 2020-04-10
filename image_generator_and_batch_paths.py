import cv2
from keras.preprocessing.image import ImageDataGenerator as kerasImageGenerator
import numpy as np
import os
from PIL import Image
import random


class ImageDataGenerator:
    def __init__(self):
        self.train_directory = ""
        self.gt_directory = ""
        self.batch_size = 0
        self.number_of_patches_per_image = 0
        self.patch_size = None

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
        self.batch_size = batch_size

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
            shuffle=True, batch_size=self.batch_size)
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
            batch_image_paths = [datagen.filepaths[idx] for idx in index_array]

            yield batch_images, batch_image_paths

    def __loadCorrespondingGtImages__(self, train_batch_paths):
        gt_batch = []
        for train_image_path in train_batch_paths:
            ground_truth_image_path = "{}{}".format(
                self.gt_directory,
                train_image_path.split(self.train_directory)[-1])
            gt_batch.append(np.array(
                Image.open(ground_truth_image_path), dtype=np.float32))
        return np.array(gt_batch)

    def __pickRandomTrainGtPatches__(self, train_batch, gt_batch):
        # Define max coordinates for image cropping
        max_x_coordinate = train_batch.shape[2] - patch_size
        max_y_coordinate = train_batch.shape[1] - patch_size

        # Take random patches from images
        train_patches = []
        gt_patches = []
        for i in range(train_batch.shape[0]):
            for j in range(self.number_of_patches_per_image):
                x_start = random.randint(0, max_x_coordinate)
                x_end = x_start + patch_size
                y_start = random.randint(0, max_y_coordinate)
                y_end = y_start + patch_size
                train_patches.append(
                    train_batch[i, y_start:y_end, x_start:x_end])
                gt_patches.append(gt_batch[i, y_start:y_end, x_start:x_end])

        return np.array(train_patches), np.array(gt_patches)

    def trainAndGtBatchGenerator(
            self, train_directory, ground_truth_directory, batch_size,
            number_of_random_patches_per_image=0, patch_size=None,
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
        number_of_random_patches_per_image : int
            If > 0 and patch_size defined, crop patches from images, must be
            less than batch_size
        patch_size : int of None
            If defined and number_of_random_patches_per_image > 0, crop patches
            from images
        normalize : bool
            If true, normalize train and ground truth arrays to range [-1, 1]

        Return
        ------
        Generator of (train_images, ground_truth_images) pair
        """
        # Set member variables
        self.train_directory = train_directory
        self.gt_directory = ground_truth_directory
        self.batch_size = batch_size
        self.number_of_patches_per_image = number_of_random_patches_per_image
        self.patch_size = patch_size

        number_of_taken_images = self.batch_size  #math.ceil(batch_size / SAMPLES_PER_IMAGE)
        batch_generator = self.batchGeneratorAndPaths(
            self.train_directory, number_of_taken_images)

        # Take respective ground truth images
        while True:
            train_batch, train_image_paths = next(batch_generator)
            gt_batch = self.__loadCorrespondingGtImages__(train_image_paths)
            if self.patch_size is not None and \
                    self.number_of_patches_per_image > 0:
                train_batch, gt_batch = self.__pickRandomTrainGtPatches__(
                    train_batch, gt_batch)
            if normalize:
                train_batch = self.normalizeArray(train_batch)
                gt_batch = self.normalizeArray(gt_batch)
            yield train_batch, gt_batch

    def normalizeArray(self, data_array, max_value=255):
        return (data_array / max_value - 0.5) * 2

    def unnormalizeArray(self, data_array, max_value=255):
        data_array = (data_array / 2 + 0.5) * max_value
        data_array[data_array < 0.0] = 0.0
        data_array[data_array > max_value] = max_value
        return data_array


def hconcatArray(data_array, resize_factor=1):
    image_concatenated = cv2.cvtColor(
        cv2.hconcat(list(data_array)) / 255, cv2.COLOR_RGB2BGR)
    new_shape = tuple(list(np.array(list(
        image_concatenated.shape[:2])) // resize_factor)[::-1])
    return cv2.resize(image_concatenated, new_shape)


def plotBatchImages(data_path, batch_size):
    data_generator = ImageDataGenerator().batchGeneratorAndPaths(
        data_path, batch_size)
    for i in range(5):
        batch_images, image_names = next(data_generator)
        print("------Batch {}------".format(i+1))
        for j in range(len(image_names)):
            print(image_names[j])
        print()
        concat_image = hconcatArray(batch_images, 8)
        cv2.imshow(str(i+1), concat_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def plotTrainAndGtBatchImages(train_data_path, gt_data_path, batch_size):
    data_generator = ImageDataGenerator().trainAndGtBatchGenerator(
        train_data_path, gt_data_path, batch_size)
    for i in range(5):
        train_batch, gt_batch = next(data_generator)
        print("Batch {}".format(i+1))
        concat_train = hconcatArray(train_batch, 8)
        concat_gt = hconcatArray(gt_batch, 8)
        concat_image = cv2.vconcat([concat_train, concat_gt])
        cv2.imshow(str(i+1), concat_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    train_data_path = "../REDS/train_blur"
    ground_truth_data_path = "../REDS/train_sharp"
    batch_size = 8
    plotBatchImages(train_data_path, batch_size)
    plotTrainAndGtBatchImages(
        train_data_path, ground_truth_data_path, batch_size)
