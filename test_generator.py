import cv2
import math
import numpy as np
from PIL import Image
import sys

from image_data_generator_2 import ImageDataGenerator


def hconcatArray(data_array, resize_factor=1):
    data_array = data_array.astype(np.float32)
    if len(data_array.shape) == 5:
        bursts = []
        for i in range(data_array.shape[0]):
            burst_concatenated = cv2.vconcat(list(data_array[i]))
            bursts.append(burst_concatenated)
        data_array = np.array(bursts)
    image_concatenated = cv2.cvtColor(
        cv2.hconcat(list(data_array)) / 255, cv2.COLOR_RGB2BGR)
    new_shape = tuple(list(np.array(list(
        image_concatenated.shape[:2])) // resize_factor)[::-1])
    return cv2.resize(image_concatenated, new_shape)


def plotBatchImages(
        data_path, batch_size, burst_size=1, max_number_of_batch_iterations=4):
    # Load data
    image_generator = ImageDataGenerator()
    data_generator = image_generator.batchGeneratorAndPaths(
        data_path, batch_size, burst_size=burst_size)
    number_of_batches_per_epoch = image_generator.numberOfBatchesPerEpoch(
        data_path, batch_size, burst_size=burst_size)

    # Loop batches
    total_number_of_batches = 0
    epochs = 0
    loop_infinite = True
    while loop_infinite:
        epochs += 1
        for i in range(number_of_batches_per_epoch):
            total_number_of_batches += 1
            batch_images, image_names = next(data_generator)
            print("------Batch {}------".format(i+1))
            for j in range(len(image_names)):
                print(image_names[j])
            print()
            concat_image = hconcatArray(batch_images, 8)
            cv2.imshow(str(i+1), concat_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if total_number_of_batches >= max_number_of_batch_iterations:
                loop_infinite = False
                break


def plotTrainAndGtBatches(
        train_data_path, gt_data_path, batch_size, patches_per_image,
        patch_size, burst_size=1, max_number_of_batch_iterations=4,
        resize_factor=4):
    # Load data
    image_generator = ImageDataGenerator()
    data_generator = image_generator.trainAndGtBatchGenerator(
        train_data_path, gt_data_path, batch_size, patches_per_image,
        patch_size, burst_size=burst_size)
    number_of_batches_per_epoch = image_generator.numberOfBatchesPerEpoch(
        train_data_path, batch_size, patches_per_image, burst_size=burst_size)

    # Loop batches
    total_number_of_batches = 0
    epochs = 0
    loop_infinite = True
    while loop_infinite:
        epochs += 1
        for i in range(number_of_batches_per_epoch):
            total_number_of_batches += 1
            train_batch, gt_batch = next(data_generator)
            if i == 0:
                print("Epoch {:2}. Batch {:2}. Images {}".format(
                    epochs, i+1, train_batch.shape[0]))
            else:
                print("          Batch {:2}. Images {}".format(
                    i+1, train_batch.shape[0]))
            concat_train = hconcatArray(train_batch, 2)
            concat_gt = hconcatArray(gt_batch, 2)
            concat_image = cv2.vconcat([concat_train, concat_gt])
            new_shape = tuple(list(np.array(list(
                concat_image.shape[:2])) // resize_factor)[::-1])
            concat_image = cv2.resize(concat_image, new_shape)
            cv2.imshow(str(i+1), concat_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if total_number_of_batches >= max_number_of_batch_iterations:
                loop_infinite = False
                break


def main():
    if len(sys.argv) <= 1:
        print("Give train data path as argument")
        return

    train_data_path = sys.argv[1]
    batch_size = 8
    patch_size = 256
    patches_per_image = 2
    burst_size = 5

    # Plot batches and print batch file names on one dataset
    plotBatchImages(train_data_path, batch_size, burst_size)

    if len(sys.argv) == 3:
        # Plot train and ground truth image pairs
        ground_truth_data_path = sys.argv[2]
        plotTrainAndGtBatches(
            train_data_path, ground_truth_data_path, batch_size, 0, None,
            burst_size=burst_size)

        # Plot train and ground truth patch pairs
        plotTrainAndGtBatches(
            train_data_path, ground_truth_data_path, batch_size,
            patches_per_image, patch_size, burst_size=burst_size,
            max_number_of_batch_iterations=20, resize_factor=1)
    else:
        print("No ground truth data path given as argument")


main()
