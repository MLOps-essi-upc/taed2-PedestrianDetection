# Dataset Card PedestrianDataset

## Table of Contents
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
    - [Data Fields](#data-fields)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Contributions](#contributions)

## Dataset Description

- **Homepage:** [https://www.kaggle.com/datasets/psvishnu/pennfudan-database-for-pedestrian-detection-zip]()

### Dataset Summary

The PennFundan dataset is an image dataset containing pictures taken from scenes around campus and urban street. More specifically, these images were captured around the University of Pennsylvania and Fundan University. This dataset was developed to aid in the task of object detection. 

The PennFundanPed dataset is a subset of the PennFundan dataset in which there is only one class of objects: PASpersonWalking, in other words, pedestrians. This dataset is specifically designed for the purpose of detecting and segmenting pedestrians in images. The dataset we will be discusing is an adaptation of the PennFundanPed dataset.

### Supported Tasks and Leaderboards

- `pedestrian_det_seg`: The dataset can be used to train a model that detects and segments pedestrians. Given an image, the model is asked to return the location of pedestrians in it using bounding boxes and masks. A bounding box is a rectangular shape that encloses an object. It is defined by the pixel coordinates of the lower-left and top-right corner, represented as [Xmin, Xmax, Ymin, Ymax]. On the other hand, a mask is an image in which pixels are assigned a value of 0 to represent the background or a value greater than 0 to indicate a specific pedestrian's ID. Success on this task is typically measured by achieving a *high* [Mean average precision and Mean average recall](https://manalelaidouni.github.io/Evaluating-Object-Detection-Models-Guide-to-Performance-Metrics.html#average-recall-ar). An [improved Mask R-CNN](https://iopscience.iop.org/article/10.1088/1742-6596/1575/1/012067/pdf) model trained to detect humans in the PennFundan dataset currently achieves a mAP of 0.835 on bounding boxes, a mAP of 0.783 on segmentation, a mAR of 0.866 on bounding boxes and a mAR of 0.813 on segmentation. 

### Languages

The text in the dataset is in English.

## Dataset Structure

### Data Instances

In order to structure the data, we've developed the 'PedestrianDataset' class. This class loads images and their corresponding annotations, including bounding boxes, class labels, masks, and other information necessary for pedestrian detection tasks.

An example from the dataset looks as follows:

```
{
  (tensor([[[0.5294, 0.3373, 0.3098,  ..., 0.5725, 0.5216, 0.4706],
          [0.3922, 0.2784, 0.3059,  ..., 0.5647, 0.5098, 0.4627],
          [0.2824, 0.2353, 0.2863,  ..., 0.5608, 0.5059, 0.4588],
          ...,
          [0.5686, 0.5686, 0.5529,  ..., 0.4627, 0.4627, 0.4627],
          [0.5608, 0.5569, 0.5490,  ..., 0.4510, 0.4510, 0.4510],
          [0.5451, 0.5490, 0.5451,  ..., 0.4471, 0.4471, 0.4471]]]),
 {'boxes': tensor([[111.,  68., 217., 345.],
          [377.,  75., 528., 376.],
          [316., 107., 346., 191.]]),
  'labels': tensor([1, 1, 1]),
  'masks': tensor([[[0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           ...,
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0]],
            ...,
          [[0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           ...,
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8),
  'image_id': tensor([6]),
  'area': tensor([29362., 45451.,  2520.]),
  'iscrowd': tensor([0, 0, 0])})
}
```

### Data Fields

- `image`: This field represents the image data. It is a 3-channel (RGB) image represented as a PIL.Image object.

- `boxes`: This field is a tensor containing the bounding box coordinates for the detected pedestrians in the image. Each bounding box is represented as a list of four values: [xmin, ymin, xmax, ymax], where (xmin, ymin) is the top-left corner, and (xmax, ymax) is the bottom-right corner of the bounding box. The data type is torch.float32.

- `labels`: This field is a tensor containing class labels for the detected pedestrians. In this dataset, there is only one class (pedestrian), so all entries in this tensor are set to 1, indicating pedestrian class. The data type is torch.int64.

- `masks`: This field represents the binary masks for each detected pedestrian instance. It is a tensor of type torch.uint8, where each mask corresponds to a unique pedestrian instance in the image. The masks are binary, with '1's indicating the pixels that belong to the pedestrian and '0's indicating the background.

- `image_id`: This field is a tensor containing a unique identifier for the image. It is typically an integer value that helps associate annotations with specific images.

- `area`: This field contains the area of each bounding box. It is computed as the product of the width and height of the bounding box: area = (xmax - xmin) * (ymax - ymin).

- `iscrowd`: This field indicates whether the instances are considered "crowded." In this dataset, all instances are assumed not to be crowded, so this tensor contains zeros.

### Data Splits

The initial dataset comprised 170 images. Following preprocessing, as we will detail in the upcoming sections, the data has been divided as follows:


|                         | train | validation |  test  |
|-------------------------|------:|-----------:|-----:  |
| data instances          |   280 |          50|     50 |
| %                       |  73.7 |       13.15|  13.15 |



## Dataset Creation

### Curation Rationale

PennFundan dataset was originally built to support research in [object detection (combining recognition and segmentation)](https://core.ac.uk/download/pdf/214172897.pdf). The dataset includes various object classes, such as pedestrian, bike, human riding bike, umbrella and car. Within this dataset, a specialized subset called PennFundanPed was created, exclusively containing the pedestrian category. This subset was designed to focus on the task of pedestrian detection which has applications in autonomous vehicles, surveillance systems, and traffic management among others.


### Source Data

#### Initial Data Collection and Normalization

The data was collected by taking photos from scenes around the campuses and urban streets of both the University of Pennsylvania and Fudan University, ensuring that each photo contained at least one pedestrian. The heights of labeled pedestrians within this database range from 180 to 390 pixels, and all of them are straight up.

As the original PennFundanPed dataset comprises only 170 images, data augmentation has been performed. More specifically, the following transformations have been carried out:

* `T.RandomHorizontalFlip(p=1)`: This transformation applies random horizontal flips to the images with a probability p of 1, meaning it flips all images horizontally. 
* `T.RandomShortestSize(120,800)`: This transformation randomly resizes the shortest side of the image to a value between 120 and 800 pixels while maintaining the aspect ratio. This can be useful for data augmentation by introducing variations in the image sizes.
* `T.RandomPhotometricDistort(p=1)`: This transformation applies random photometric distortions to the images with a probability p of 1, meaning it is always applied. Photometric distortions include changes in brightness, contrast, saturation, and hue, introducing variations in image appearance.

It's important to note that data augmentation is selectively applied only during the training phase, ensuring that our validation and testing datasets remain unaltered and maintain their original characteristics. During training, we create multiple training datasets, each incorporating distinct augmentation transforms. By following this approach, we can effectively assess our model's performance on unseen data while leveraging data augmentation to enhance its training process.

#### Who are the producers?

The identity of the photographer remains undisclosed. However, the PennFundan dataset was built by the authors of the article [Object detection combining Recognition and Segmentation](https://core.ac.uk/download/pdf/214172897.pdf).

### Annotations

The dataset does not contain any additional annotations.

### Personal and Sensitive Information

Due to the fact that the database contains photos on the street, it includes personal and sensitive information. First, there are the facial features captured in these images, which have the potential to reveal the identity of individuals. Moreover, the images feature vehicles with the license plates visible, providing sensitive information that can be linked to vehicle owners.

## Considerations for Using the Data

### Social Impact of Dataset

The purpouse of this dataset is to help develop models that can detect pedestrians. Such systems can be:
- Integrated into autonomous vehicles to enhance road safety and reduce the number of pedestrian-involved accidents
- Used to make informed decisions about trafic flow and urban planning 
- Used for surveillance and security purposes  

However, these systems can raise concerns about individual privacy, as it may enable the tracking of people's movements without their consent.

### Discussion of Biases

Without a balanced dataset, machine learning models can become biased towards the majority. This training data primarily consists of one race, asians, so the model may not perform well when encountering underrepresented races, leading to accuracy and fairness issues. For instance, in authonomous driving failing to detect pedestrians from certain races can result in accidents or dangerous situations.

Recent research has addressed the issue of racial biases in machine learning systems, as demonstrated in studies like "[Predictive Inequity in Object Detection](https://arxiv.org/pdf/1902.11097.pdf)". These studies have revealed a consistent pattern that AI systems exhibit higher accuracy in identifying pedestrians with lighter skin tones compared to those with darker skin tones.

### Other Known Limitations

The dataset's restriction of pedestrian heights between 180 and 390 pixels may not cover extreme cases or unusual situations, potentially leading to inaccuracies in pedestrian detection for individuals outside this range. 

In the same way, limiting the dataset to pedestrians in an upright position may not account for real-world scenarios where pedestrians have different body postures, which can affect the model's robustness.

However, these two limitations have been reduced thanks to the data augmentation transforms applied.

Objects that are severely occluded or very small in size may not be included in the ground truth annotations. This omission can affect the model's ability to detect all relevant objects.

## Additional Information 

### Dataset Curators

List the people involved in collecting the dataset and their affiliation(s). If funding information is known, include it here.

Rodrigo Bonferroni, Arlet Corominas and Clàudia Mur.

### Contributions

Thanks to [@psvishnu-kaggle]([https://github.com/<github-username>](https://www.kaggle.com/psvishnu)https://www.kaggle.com/psvishnu) for adding this dataset.
