---
YAML tags (full spec here: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1):
- copy-paste the tags obtained with the online tagging app: https://huggingface.co/spaces/huggingface/datasets-tagging
---

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
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)

## Dataset Description

- **Homepage:** [https://www.kaggle.com/datasets/psvishnu/pennfudan-database-for-pedestrian-detection-zip]()
- **Paper:** [If the dataset was introduced by a paper or there was a paper written describing the dataset, add URL here (landing page for Arxiv paper preferred)]()
- **Leaderboard:** [If the dataset supports an active leaderboard, add link here]()
- **Point of Contact:** [If known, name and email of at least one person the reader can contact for questions about the dataset.]()

[https://www.kaggle.com/datasets/psvishnu/pennfudan-database-for-pedestrian-detection-zip]()

*NOMÉS DEIXARIA EL PRIMER

### Dataset Summary

<span style="color:grey">Briefly summarize the dataset, its intended use and the supported tasks. Give an overview of how and why the dataset was created. The summary should explicitly mention the languages present in the dataset (possibly in broad terms, e.g. *translations between several pairs of European languages*), and describe the domain, topic, or genre covered.</span>

The PennFundan dataset is an image dataset containing pictures taken from scenes around campus and urban street. More specifically, these images were captured around the University of Pennsylvania and Fundan University. This dataset was developed to aid in the task of object detection. 

The PennFundanPed dataset is a subset of the PennFundan dataset in which there is only one class of objects: PASpersonWalking, in other words, pedestrians. This dataset is specifically designed for the purpose of detecting and segmenting pedestrians in images. The dataset we will be discusing is an adaptation of the PennFundanPed dataset.


### Supported Tasks and Leaderboards

<span style="color:grey">For each of the tasks tagged for this dataset, give a brief description of the tag, metrics, and suggested models (with a link to their HuggingFace implementation if available). Give a similar description of tasks that were not covered by the structured tag set (repace the `task-category-tag` with an appropriate `other:other-task-name`).</span>

- `pedestrian_det_seg`: The dataset can be used to train a model that detects and segments pedestrians. Given an image, the model is asked to return the location of pedestrians in it using bounding boxes and masks. A bounding box is a rectangular shape that encloses an object. It is defined by the pixel coordinates of the lower-left and top-right corner, represented as [Xmin, Xmax, Ymin, Ymax]. On the other hand, a mask is an image in which pixels are assigned a value of 0 to represent the background or a value greater than 0 to indicate a specific pedestrian's ID. Success on this task is typically measured by achieving a *high* [Mean average precision and Mean average recall](https://manalelaidouni.github.io/Evaluating-Object-Detection-Models-Guide-to-Performance-Metrics.html#average-recall-ar). An [improved Mask R-CNN](https://iopscience.iop.org/article/10.1088/1742-6596/1575/1/012067/pdf
) model trained to detect humans in the PennFundan dataset currently achieves a mAP of 0.835 on bounding boxes, a mAP of 0.783 on segmentation, a mAR of 0.866 on bounding boxes and a mAR of 0.813 on segmentation. 


### Languages

<span style="color:grey">Provide a brief overview of the languages represented in the dataset. Describe relevant details about specifics of the language such as whether it is social media text, African American English,...
When relevant, please provide [BCP-47 codes](https://tools.ietf.org/html/bcp47), which consist of a [primary language subtag](https://tools.ietf.org/html/bcp47#section-2.2.1), with a [script subtag](https://tools.ietf.org/html/bcp47#section-2.2.3) and/or [region subtag](https://tools.ietf.org/html/bcp47#section-2.2.4) if available.</span>

The text in the dataset is in English.


## Dataset Structure

### Data Instances

<span style="color:grey">Provide an JSON-formatted example and brief description of a typical instance in the dataset. If available, provide a link to further examples.</span>

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

<span style="color:grey">Provide any additional information that is not covered in the other sections about the data here. In particular describe any relationships between data points and if these relationships are made explicit.</span>



### Data Fields

<span style="color:grey">List and describe the fields present in the dataset. Mention their data type, and whether they are used as input or output in any of the tasks the dataset currently supports. If the data has span indices, describe their attributes, such as whether they are at the character level or word level, whether they are contiguous or not, etc. If the datasets contains example IDs, state whether they have an inherent meaning, such as a mapping to other datasets or pointing to relationships between data points.</span>


- `image`: This field represents the image data. It is a 3-channel (RGB) image represented as a PIL.Image object.

- `boxes`: This field is a tensor containing the bounding box coordinates for the detected pedestrians in the image. Each bounding box is represented as a list of four values: [xmin, ymin, xmax, ymax], where (xmin, ymin) is the top-left corner, and (xmax, ymax) is the bottom-right corner of the bounding box. The data type is torch.float32.

- `labels`: This field is a tensor containing class labels for the detected pedestrians. In this dataset, there is only one class (pedestrian), so all entries in this tensor are set to 1, indicating pedestrian class. The data type is torch.int64.

- `masks`: This field represents the binary masks for each detected pedestrian instance. It is a tensor of type torch.uint8, where each mask corresponds to a unique pedestrian instance in the image. The masks are binary, with '1's indicating the pixels that belong to the pedestrian and '0's indicating the background.

- `image_id`: This field is a tensor containing a unique identifier for the image. It is typically an integer value that helps associate annotations with specific images.

- `area`: This field contains the area of each bounding box. It is computed as the product of the width and height of the bounding box: area = (xmax - xmin) * (ymax - ymin).

- `iscrowd`: This field indicates whether the instances are considered "crowded." In this dataset, all instances are assumed not to be crowded, so this tensor contains zeros.


<span style="color:grey">Note that the descriptions can be initialized with the **Show Markdown Data Fields** output of the [Datasets Tagging app](https://huggingface.co/spaces/huggingface/datasets-tagging), you will then only need to refine the generated descriptions.</span>


### Data Splits

<span style="color:grey">Describe and name the splits in the dataset if there are more than one.</span>

<span style="color:grey">Describe any criteria for splitting the data, if used. If there are differences between the splits (e.g. if the training annotations are machine-generated and the dev and test ones are created by humans, or if different numbers of annotators contributed to each example), describe them here.</span>

<span style="color:grey">Provide the sizes of each split. As appropriate, provide any descriptive statistics for the features, such as average length.  For example:</span>

The initial dataset comprised 170 images. Following preprocessing, as we will detail in the upcoming sections, the data has been divided as follows:


|                         | train | validation |  test  |
|-------------------------|------:|-----------:|-----:  |
| data instances          |   280 |          50|     50 |
| %                       |  73.7 |       13.15|  13.15 |



## Dataset Creation

### Curation Rationale

<span style="color:grey">What need motivated the creation of this dataset? What are some of the reasons underlying the major choices involved in putting it together?</span>

PennFundan dataset was originally built to support research in [object detection (combining recognition and segmentation)](https://core.ac.uk/download/pdf/214172897.pdf). The dataset includes various object classes, such as pedestrian, bike, human riding bike, umbrella and car. Within this dataset, a specialized subset called PennFundanPed was created, exclusively containing the pedestrian category. This subset was designed to focus on the task of pedestrian detection which has applications in autonomous vehicles, surveillance systems, and traffic management among others.


### Source Data

<span style="color:grey">This section describes the source data (e.g. news text and headlines, social media posts, translated sentences,...)</span>

#### Initial Data Collection and Normalization

<span style="color:grey">Describe the data collection process. Describe any criteria for data selection or filtering. List any key words or search terms used. If possible, include runtime information for the collection process.</span>

<span style="color:grey">If data was collected from other pre-existing datasets, link to source here and to their [Hugging Face version](https://huggingface.co/datasets/dataset_name).</span>

<span style="color:grey">If the data was modified or normalized after being collected (e.g. if the data is word-tokenized), describe the process and the tools used.</span>

The data was collected by taking photos from scenes around the campuses and urban streets of both the University of Pennsylvania and Fudan University, ensuring that each photo contained at least one pedestrian. The heights of labeled pedestrians within this database range from 180 to 390 pixels, and all of them are straight up.

As the original PennFundanPed dataset comprises only 170 images, data augmentation has been performed. More specifically, the following transformations have been carried out:

* `T.RandomHorizontalFlip(p=1)`: This transformation applies random horizontal flips to the images with a probability p of 1, meaning it flips all images horizontally. 
* `T.RandomShortestSize(120,800)`: This transformation randomly resizes the shortest side of the image to a value between 120 and 800 pixels while maintaining the aspect ratio. This can be useful for data augmentation by introducing variations in the image sizes.
* `T.RandomPhotometricDistort(p=1)`: This transformation applies random photometric distortions to the images with a probability p of 1, meaning it is always applied. Photometric distortions include changes in brightness, contrast, saturation, and hue, introducing variations in image appearance.

It's important to note that data augmentation is selectively applied only during the training phase, ensuring that our validation and testing datasets remain unaltered and maintain their original characteristics. During training, we create multiple training datasets, each incorporating distinct augmentation transforms. By following this approach, we can effectively assess our model's performance on unseen data while leveraging data augmentation to enhance its training process.




#### Who are the producers?

<span style="color:grey">State whether the data was produced by humans or machine generated. Describe the people or systems who originally created the data.

<span style="color:grey">If available, include self-reported demographic or identity information for the source data creators, but avoid inferring this information. Instead state that this information is unknown. See [Larson 2017](https://www.aclweb.org/anthology/W17-1601.pdf) for using identity categories as a variables, particularly gender.

<span style="color:grey">Describe the conditions under which the data was created (for example, if the producers were crowdworkers, state what platform was used, or if the data was found, what website the data was found on). If compensation was provided, include that information here.

<span style="color:grey">Describe other people represented or mentioned in the data. Where possible, link to references for the information.

The identity of the photographer remains undisclosed. However, the PennFundan dataset was built by the authors of the article [Object detection combining Recognition and Segmentation](https://core.ac.uk/download/pdf/214172897.pdf).

NO SE SI PUC ELIMINAR LO DELS AUTORS I NMS M'ESTÀ PREGUNTANT PER LES FOTOS*

### Annotations

<span style="color:grey">If the dataset contains annotations which are not part of the initial data collection, describe them in the following paragraphs.

The dataset does not contain any additional annotations.

#### Annotation process

<span style="color:grey">If applicable, describe the annotation process and any tools used, or state otherwise. Describe the amount of data annotated, if not all. Describe or reference annotation guidelines provided to the annotators. If available, provide interannotator statistics. Describe any annotation validation processes.

[N/A]

#### Who are the annotators?

<span style="color:grey">If annotations were collected for the source data (such as class labels or syntactic parses), state whether the annotations were produced by humans or machine generated.

<span style="color:grey">Describe the people or systems who originally created the annotations and their selection criteria if applicable.

<span style="color:grey">If available, include self-reported demographic or identity information for the annotators, but avoid inferring this information. Instead state that this information is unknown. See [Larson 2017](https://www.aclweb.org/anthology/W17-1601.pdf) for using identity categories as a variables, particularly gender.

<span style="color:grey">Describe the conditions under which the data was annotated (for example, if the annotators were crowdworkers, state what platform was used, or if the data was found, what website the data was found on). If compensation was provided, include that information here.

[N/A]

### Personal and Sensitive Information

<span style="color:grey">State whether the dataset uses identity categories and, if so, how the information is used. Describe where this information comes from (i.e. self-reporting, collecting from profiles, inferring, etc.). See [Larson 2017](https://www.aclweb.org/anthology/W17-1601.pdf) for using identity categories as a variables, particularly gender. State whether the data is linked to individuals and whether those individuals can be identified in the dataset, either directly or indirectly (i.e., in combination with other data).

<span style="color:grey">State whether the dataset contains other data that might be considered sensitive (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history).  

<span style="color:grey">If efforts were made to anonymize the data, describe the anonymization process.

Due to the fact that the database contains photos on the street, it includes personal and sensitive information. First, there are the facial features captured in these images, which have the potential to reveal the identity of individuals. Moreover, the images feature vehicles with the license plates visible, providing sensitive information that can be linked to vehicle owners.

## Considerations for Using the Data

### Social Impact of Dataset

<span style="color:grey">Please discuss some of the ways you believe the use of this dataset will impact society.

<span style="color:grey">The statement should include both positive outlooks, such as outlining how technologies developed through its use may improve people's lives, and discuss the accompanying risks. These risks may range from making important decisions more opaque to people who are affected by the technology, to reinforcing existing harmful biases (whose specifics should be discussed in the next section), among other considerations.

<span style="color:grey">Also describe in this section if the proposed dataset contains a low-resource or under-represented language. If this is the case or if this task has any impact on underserved communities, please elaborate here.

The purpouse of this dataset is to help develop models that can detect pedestrians. Such systems can be:
- Integrated into autonomous vehicles to enhance road safety and reduce the number of pedestrian-involved accidents
- Used to make informed decisions about trafic flow and urban planning 
- Used for surveillance and security purposes  

However, these systems can raise concerns about individual privacy, as it may enable the tracking of people's movements without their consent.


### Discussion of Biases

<span style="color:grey">Provide descriptions of specific biases that are likely to be reflected in the data, and state whether any steps were taken to reduce their impact.

<span style="color:grey">For Wikipedia text, see for example [Dinan et al 2020 on biases in Wikipedia (esp. Table 1)](https://arxiv.org/abs/2005.00614), or [Blodgett et al 2020](https://www.aclweb.org/anthology/2020.acl-main.485/) for a more general discussion of the topic.

<span style="color:grey">If analyses have been run quantifying these biases, please add brief summaries and links to the studies here.

Without a balanced dataset, machine learning models can become biased towards the majority. This training data primarily consists of one race, asians, so the model may not perform well when encountering underrepresented races, leading to accuracy and fairness issues. For instance, in authonomous driving failing to detect pedestrians from certain races can result in accidents or dangerous situations.

Recent research has addressed the issue of racial biases in machine learning systems, as demonstrated in studies like "[Predictive Inequity in Object Detection](https://arxiv.org/pdf/1902.11097.pdf)". These studies have revealed a consistent pattern that AI systems exhibit higher accuracy in identifying pedestrians with lighter skin tones compared to those with darker skin tones.


### Other Known Limitations

<span style="color:grey">If studies of the datasets have outlined other limitations of the dataset, such as annotation artifacts, please outline and cite them here.

The dataset's restriction of pedestrian heights between 180 and 390 pixels may not cover extreme cases or unusual situations, potentially leading to inaccuracies in pedestrian detection for individuals outside this range. 

In the same way, limiting the dataset to pedestrians in an upright position may not account for real-world scenarios where pedestrians have different body postures, which can affect the model's robustness.

However, these two limitations have been reduced thanks to the data augmentation transforms applied.

Objects that are severely occluded or very small in size may not be included in the ground truth annotations. This omission can affect the model's ability to detect all relevant objects.

## Additional Information 

### Dataset Curators

List the people involved in collecting the dataset and their affiliation(s). If funding information is known, include it here.

Rodrigo Bonferroni, Arlet Corominas and Clàudia Mur.


### Contributions

Thanks to [@github-username](https://github.com/<github-username>) for adding this dataset.