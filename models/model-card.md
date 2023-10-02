
# Model Card for MASK R-CNN ResNet-50

Mask R-CNN extends the model Faster R-CNN to solve instance segmentation tasks. It can not only tell you what objects are in an image but also exactly where each object is located and which pixels belong to it, because is able to generates precise pixel-level mask for each detect object. It's widely used in various applications, but in our case, is designed for pedestrian detection. 

More over, a ResNet architecture is implemented to overcome the vanishing gradient problem and improve the training of deep neural networks by using skip connections within each residual blocks. 

## Model Details

### Model Description

- **Developed by:** Group of researchers at Facebook AI Research (FAIR) including Kaiming He, Georgia Gkioxari, Piotr Gollár and Ross Girshick, among others.

- **Model type:** Mask R-CNN is a deep learning model that belongs to the family of Convolutional Neural Networks (CNNs). It is specifically designed for the task of instance segmentation, which combines object detection (identifying objects and their bounding boxes) and semantic segmentation (pixel-wise object labeling, masks). 

- **Language(s) (NLP):** It is typically implemented in Python using deep learning frameworks such as TensorFlow or PyTorch.

- **Finetuned from model:** The model base [model base](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html) used is the one found in PyTorch.

## Uses
### Direct Use

Direct use of the model involves training the model on our dataset to ahieve instance segmentation in pedestrian detection. So, it is able to identify objects in images while providing pixel-level masks for eash instance of the object.

### Out-of-Scope Use

This model may not be suitable for detect another class different form persons.

## Bias, Risks, and Limitations

The ***biases*** in the model are typically a reflection of the biases present in the training data. In this case, we have only one type of images; where one or two persons are in a vertial position, walking, with asiatic characteristics and a determine stature.  

These characteristics may lead to some potential ***risks*** and ***limitations*** associated with the model:

- **Data Requirement - Loss of information:** It is important to have a quality of data, so here you could have insufficent images and a poor sample of the population.

- **Computational Intensity:** The model is computationally intensive and may require powerful hardware (GPUs) for real-time applications.

- **Limited to 2D:** The model operates in 2D space and may not be suitable for tasks that require 3D understanding.

### Recommendations

Users should be made aware of the risks, biases and limitations of the model. 

## How to Get Started with the Model

Use the code below to get started with the model. In order for the code to work, its important to have the data and the code uploaded in google drive, this way you could connect the code to your content drive. Moreover, some package need to be installed, as Python, PyTorhc, NumPy, pickle. ML flow is used to but during the code ou will find the command to install it.

## Training Details

### Training Data

The dataset used is the *PenFundanPed Dataset*, which we have preprocessed by applying data augmentation and structuring it based on the `DatasetPedestrian` class. This has allowed us to have images for training, validation, and testing. For the training images, we applied three transformations, and their information is organized with various fields such as `image`, `boxes`, `labels`, `masks`, `image_id`, etc.

Al this information, and more, is witting in the [Dataset Card](https://github.com/MLOps-essi-upc/taed2-PedestrianDetection/blob/main/data/dataset-card.md) below.

### Training Procedure 

This model was constructed based on the Mask R-CNN ResNet-50 model, which was pretained to detect objects in the popular Coco dataset. But we have performed different experiments to choose the best model for our practice. So, we first trained using the model Mask R-CNN ResNet-50 of the shelf and then with fine-tuning, where we had to modified the architecture of the bounding box and the mask predictor. This way, during the fine-tune, we were able to change some hyperparameters to search for the best performance.

#### Training Hyperparameters

The model baseline have especifics settings that, during the fine-tuning we change it with different values:

|                         | Baseline | Fine-tune  |
|-------------------------|---------:|-----------:|
| Size hidden layer       |   216    |  128 | 512 |
| Batch size              |  2       | 4    | 8   |
| Number of epochs        | 3        |      2     |

## Evaluation

For each one of the different hyperparameters, and evaluation of the model have ben done, generating different metrics to uderstand the performance.

#### Metrics

- **Average recall (AR):** (for the mask and bounding boxes) evaluates how well the model detects and recalls correct ojects, measuring the proportion of the true positive predictions that the model correctly identifies.

- **Average precisions (AP):** (for the mask and bounding boxes) quantifies how well the model precisely identifies and localizes objects within the image, considering both the accuracy and spatial overlap of the predictions.

- **Loss:** general loss metric that likely represents the overall loss of the model. It is the combination of the folowwing losses.

- **Loss classifier:** measures the loss assoceiated with the classification task in the model, quantifing how well the model is classifying instances.

- **Loss box regression:** loss related to the bounding box regression task, measuring how accurately the model is predicting the coordinates of object bounding boxes.

- **Loss mask:** related to the quality of the predicted segmentation masks, quantifing how well the model is segmenting instances within images.

- **Loss objectness:** loss associated with objectness prediction to determine whether an object is present in a specific region of an image.

- **Loss RPN box regression:** quantifies the loss for region proposal network (RPN) bounding box regression, measuring how well the model is proposing object regions.

### Results

We will be able to write this part when we are able to finish all the experiments and choose the best model in terms of AR and AP scores.

So the model, having the following characterstics (model settings):
POSAR LA MILLOR COMBINACIÓ D'HIPERPARÀMETRES VISTA

Achieved the following results (model metrics of the last epoch):
POSAR RESULTATS DEL MODEL BO

## Model Examination

<!-- Relevant interpretability work for the model goes here -->

With milestone 3 we will be able to fill this part.

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

With milestone 3 we will be able to fill this part.

## Technical Specifications [optional]

### Model Architecture and Objective

Mask R-CNN is a two stage structure that combines the architecture of Faster R-CNN, which is primarily designed for object detection, with additional components for pixel-level mask prediction. It achieves this by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition of the Faster R-CNN model. Moreover, to fix the misalignment utilises a simple, quantization-free layer, called RoIAlign, that faithfully preserves exact spatial locations.

Here's an overview of its key components:

1) **Backbone Network:** It is used a ResNet-50 to extracks features from the input image.

2) **Region Proposal Network (RPN):** Similar to Faster R-CNN, this model employs an RPN to propose regions of interest (ROIs) where objects might be present. These ROIs are then passed to subsequent stages for further processing.

3) **ROI Align:** Uses ROI Align to extract features from the ROIs. ROI Align is more precise in aligning pixel-level features, which is crucial for generating accurate instance masks.

4) **Mask Head:** This component is responsible for generating the pixel-level masks for each detected object. It typically consists of a stack of convolutional layers that predict a binary mask for each ROI (1 if there is the object below and 0 otherwise).

5) **Object Detection Head:** In addition to the mask prediction, the model also includes an object detection head that predicts the class labels and bounding box coordinates for the detected objects, with the same structure of the model Faster R-CNN.

![Model's structure schema](https://github.com/MLOps-essi-upc/taed2-PedestrianDetection/blob/main/models/Structure%20of%20model%20Mask%20R-CNN.png)

In addition, in this model there are 5 types of losses distinguished, depending on the stage you are in:

- *Regression loss:* to determine if a bounding box is proposed or not proposed.

- *Classification loss:* to determine if an object is present or not within a bounding box.

- *Regression loss:* to define the bounding box's value based on a proposal.

- *Classification loss:* to predict the class of the predicted bounding box.

- *Mask prediction loss:* to generates the pixel-level mask for each object, in binary form (1 belongs to the object, 0 otherwise).

## Model Card Contact

Rodrigo Bonferroni (@RodrigoBonferroni), Arlet Corominas (@arletcoro) and Clàudia Mur (@claudiamur).


