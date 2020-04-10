from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image


def batchGeneratorAndPaths(data_directory, batch_size, image_size=None):
    """
    Get image generator and filenames for the images in the batch

    Arguments
    --------
    data_directory : str
        Path to the data directory, directory must include folders and images
        images inside those folders
    batch_size : int
        Number of images in a batch
    image_size : tuple, None
        Target image size of what will be the resized images in a batch, for
        example (256, 256). If image_size=None, the images will be original
        sized as in the folders.

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
    datagen = ImageDataGenerator().flow_from_directory(
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
                    0, datagen.samples - datagen.samples % datagen.batch_size)
            else:
                batch_index = max(0, datagen.samples - datagen.batch_size)
        index_array = datagen.index_array[
            batch_index:batch_index + datagen.batch_size].tolist()
        batch_image_paths = [datagen.filepaths[idx] for idx in index_array]

        yield batch_images, batch_image_paths


def plotBatchImages(data_path, batch_size):
    import cv2
    data_generator = batchGeneratorAndPaths(data_path, batch_size)
    for i in range(5):
        batch_images, image_names = next(data_generator)
        print("------{}------".format(i+1))
        for j in range(len(image_names)):
            print(image_names[j])
        print()
        image_concatenated = cv2.cvtColor(
            cv2.hconcat(list(batch_images)) / 255, cv2.COLOR_RGB2BGR)
        cv2.imshow(str(i+1), image_concatenated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    data_path = "../REDS/train_blur"
    batch_size = 4
    plotBatchImages(data_path, batch_size)
