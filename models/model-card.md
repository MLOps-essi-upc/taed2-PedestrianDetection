---
'NULL': null
output:
  html_document:
    df_print: paged
---

# Model Card for MASK R-CNN ResNet-50

Mask R-CNN extends the model Faster R-CNN to solve instance segmentation tasks. It can not only tell you what objects are in an image but also exactly where each object is located and which pixels belong to it, because is able to generates precise pixel-level mask for each detect object. It's widely used in various applications, but in our case, is designed for pedestrian detection. 

More over, a ResNet architecture is implemented to overcome the vanishing gradient problem and improve the training of deep neural networks by using skip connections within each residual blocks. 

## Model Details

### Model Description

- **Developed by:**

Group of researchers at Facebook AI Research (FAIR) including Kaiming He, Georgia Gkioxari, Piotr Gollár and Ross Girshick, among others.

- **Model type:**

Mask R-CNN is a deep learning model that belongs to the family of Convolutional Neural Networks (CNNs). It is specifically designed for the task of instance segmentation, which combines object detection (identifying objects and their bounding boxes) and semantic segmentation (pixel-wise object labeling, masks). 

- **Language(s) (NLP):**

It is typically implemented in Python using deep learning frameworks such as TensorFlow or PyTorch.

- **Finetuned from model:**

The model base [model base]("https://pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html") used is the one found in PyTorch.

## Uses

Mask R-CNN has a wide range of applications, but in this case is used for:

- **Object Instance Segmentation:** To segment and identify objects in images while providing pixel-level masks for each instance of the object.

### Direct Use

Direct use of the model involves training the model on our dataset to ahieve instance segmentation in pedestrian detection.

### Out-of-Scope Use

This model may not be suitable for detect another class different form persons.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

{{ bias_risks_limitations | default("[More Information Needed]", true)}}

The ***biases*** in the model are typically a reflection of the biases present in the training data. If the training data is biased towards certain classes or underrepresents others, the model may exhibit biased behavior in its predictions.

Some ***risks*** and ***limitations*** associated with Mask R-CNN include:

- **Computational Intensity:** Mask R-CNN is computationally intensive and may require powerful hardware (GPUs) for real-time applications.

- **Data Requirement:** Training Mask R-CNN effectively often requires large annotated datasets, which may be costly and time-consuming to create.

- **Overfitting:** Like other deep learning models, Mask R-CNN can overfit to the training data if not properly regularized or if the dataset is small.

- **Limited to 2D:** Mask R-CNN operates in 2D space and may not be suitable for tasks that require 3D understanding.

- **Accuracy Trade-offs:** There may be a trade-off between accuracy and speed when using Mask R-CNN, depending on the application's requirements.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

{{ bias_recommendations | default("Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.", true)}}

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}}

## Training Details

### Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

{{ training_data | default("[More Information Needed]", true)}}

EXPLICAR AQUÍ EL DATASET (mirar dataset card)

### Training Procedure 

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

{{ preprocessing | default("[More Information Needed]", true)}}


#### Training Hyperparameters

- **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Data Card if possible. -->

{{ testing_data | default("[More Information Needed]", true)}}

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

{{ testing_metrics | default("[More Information Needed]", true)}}

### Results

{{ results | default("[More Information Needed]", true)}}

#### Summary

{{ results_summary | default("", true) }}

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

{{ model_examination | default("[More Information Needed]", true)}}

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** {{ hardware | default("[More Information Needed]", true)}}
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}}
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}

## Technical Specifications [optional]

### Model Architecture and Objective

{{ model_specs | default("[More Information Needed]", true)}}

It achieves this by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition of the Faster R-CNN model. Moreover, to fix the misalignment utilises a simple, quantization-free layer, called RoIAlign, that faithfully preserves exact spatial locations.

Mask R-CNN is a two stage structure that combines the architecture of Faster R-CNN, which is primarily designed for object detection, with additional components for pixel-level mask prediction. Here's an overview of its key components:

1) **Backbone Network:** Mask R-CNN typically uses a convolutional neural network (CNN) as its backbone, such as ResNet or ResNeXt. This backbone extracts features from the input image.

2) **Region Proposal Network (RPN):** Similar to Faster R-CNN, Mask R-CNN employs an RPN to propose regions of interest (ROIs) where objects might be present. These ROIs are then passed to subsequent stages for further processing.

3) **ROI Align:** Instead of the ROI pooling used in Faster R-CNN, Mask R-CNN uses ROI Align to extract features from the ROIs. ROI Align is more precise in aligning pixel-level features, which is crucial for generating accurate instance masks.

4) **Mask Head:** This component is responsible for generating the pixel-level masks for each detected object. It typically consists of a stack of convolutional layers that predict a binary mask for each ROI (1 if there is the object below and 0 otherwise).

5) **Object Detection Head:** In addition to the mask prediction, Mask R-CNN also includes an object detection head that predicts the class labels and bounding box coordinates for the detected objects, with the same structure of the model Faster R-CNN.

ES POT INTENTAR POSAR UNA IMATGE DE L'ARQUITECTURA

In addition, in this model there are 5 types of losses distinguished, depending on the stage we are in:

- *Regression loss:* to determine if a bounding box is proposed or not proposed.

- *Classification loss:* to determine if an object is present or not within a bounding box.

- *Regression loss:* to define the bounding box's value based on a proposal.

- *Classification loss:* to predict the class of the predicted bounding box.

- *Mask prediction loss:* to generates the pixel-level mask for each object, in binary form (1 belongs to the object, 0 otherwise).

Finally, as we already said in previous sections, the primary ***objectives*** of Mask R-CNN are:

- **Object Detection:** Identify and locate objects in an image with bounding boxes and class labels.

- **Instance Segmentation:** Generate pixel-level masks for each individual object instance.

- **Semantic Segmentation (Optional):** Some variants of Mask R-CNN can also be used for semantic segmentation, where each pixel is labeled with a category. 

So all of this makes this model a versatile tool in computer vision applications.

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

SUPOSO QUE ESTÀ TOT A CONTINUACIÓ I AQUÍ NO HE DE POSAR RES

#### Hardware

{{ hardware | default("[More Information Needed]", true)}}

Mask R-CNN, being a deep neural network, benefits from powerful hardware, often including:

- *Graphics Processing Units (GPUs):* High-end GPUs like NVIDIA's Tesla or GeForce series are commonly used to accelerate training and inference.

- *Multi-GPU setups:* Training Mask R-CNN on large datasets can be significantly accelerated using multiple GPUs in parallel.

- *Cloud Computing:* Many researchers and practitioners use cloud-based platforms (e.g., AWS, GCP, Azure) to access GPU resources on-demand.

#### Software

{{ software | default("[More Information Needed]", true)}}

The software aspects that are most commonly used for this model are:

- *Deep Learning Frameworks:* Mask R-CNN is implemented using deep learning frameworks such as TensorFlow or PyTorch.

- *Python:* The model code and its associated libraries are typically written in Python.

- *CUDA:* CUDA is a parallel computing platform and API developed by NVIDIA for GPU acceleration. It's often used in conjunction with deep learning frameworks to speed up computation on GPUs.

- *Datasets:* Common datasets used for training Mask R-CNN include COCO (Common Objects in Context), Pascal VOC, and custom datasets tailored to specific applications.

- *Preprocessing and Postprocessing Code:* Image preprocessing and postprocessing code is often used to prepare input data and interpret model outputs.

- *Evaluation Metrics:* Metrics such as Intersection over Union (IoU) and mean Average Precision (mAP) are commonly used to evaluate the performance of Mask R-CNN models.

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

{{ citation_bibtex | default("[More Information Needed]", true)}}

**APA:**

{{ citation_apa | default("[More Information Needed]", true)}}

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

{{ glossary | default("[More Information Needed]", true)}}

## More Information [optional]

{{ more_information | default("[More Information Needed]", true)}}

## Model Card Authors [optional]

{{ model_card_authors | default("[More Information Needed]", true)}}

## Model Card Contact

{{ model_card_contact | default("[More Information Needed]", true)}}

NO SÉ QUÈ S'HA DE POSAR:
You can include any additional information or context that may be relevant, such as:

The specific implementation or variant of Mask R-CNN you are using (if applicable).
The purpose or application for which you are using Mask R-CNN.
Any modifications or adaptations you've made to the model for your specific project.
The dataset(s) you used for training or fine-tuning the model.
Make sure to provide accurate and up-to-date contact information in case there are inquiries or collaborations related to your use of Mask R-CNN.


