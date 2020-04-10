import cv2
import math
import numpy as np
from PIL import Image

from image_data_generator import ImageDataGenerator


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


def plotTrainAndGtBatchPatches(
        train_data_path, gt_data_path, batch_size, patches_per_image,
        patch_size):
    data_generator = ImageDataGenerator().trainAndGtBatchGenerator(
        train_data_path, gt_data_path, batch_size, patches_per_image,
        patch_size)
    for i in range(5):
        train_batch, gt_batch = next(data_generator)
        print("Batch {}".format(i+1))
        concat_train = hconcatArray(train_batch, 2)
        concat_gt = hconcatArray(gt_batch, 2)
        concat_image = cv2.vconcat([concat_train, concat_gt])
        cv2.imshow(str(i+1), concat_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    train_data_path = "../REDS/train_blur"
    ground_truth_data_path = "../REDS/train_sharp"
    batch_size = 8
    patch_size = 256
    patches_per_image = 2

    # Plot batches on one dataset (for example test)
    plotBatchImages(train_data_path, batch_size)

    # Plot train and ground truth image pairs
    plotTrainAndGtBatchImages(
        train_data_path, ground_truth_data_path, batch_size)

    # Plot train and ground truth patch pairs
    plotTrainAndGtBatchPatches(
        train_data_path, ground_truth_data_path, batch_size, patches_per_image,
        patch_size)


main()
