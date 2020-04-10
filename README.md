# Keras batch generator and filenames

## Introduction

General image batch generator using Keras. The generator returns batch of images and image file names in a batch. The data set must be in a following format:

```bash
data_folder/
    image_folder_1/
        image_1_1.png
        image_1_2.png
        image_1_3.png
        ...
        image_1_n.png
    image_folder_2/
        image_2_1.png
        image_2_2.png
        image_2_3.png
        ...
        image_2_n.png
    image_folder_3/
        image_3_1.png
        image_3_2.png
        image_3_3.png
        ...
        image_3_n.png
    .
    .
    .
    image_folder_m/
        image_m_1.png
        image_m_2.png
        image_m_3.png
        ...
        image_m_n.png
```
