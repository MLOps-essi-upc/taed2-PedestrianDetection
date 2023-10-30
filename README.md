TAED2 - Pedestrian Detection
==============================

Pedestrain Detection is an important aspect in various domains as autonomous vehicles, surveillance systems and smart cities. For this reason, in this project, we are interested in enhances our system's accuracy by distinguishing individual pedestrains while using instance segmentation technique, one of the most advanced computer vision task. 

This project emphasizes the integration of ML into software engineering to increase its reliability, scalability and maintainability. So we will cover ML techniques for data preprocessing and model evaluation while also addressing software engineering aspects like version control, documentation and continuous integration.

All of this is supervised by the subject "Advanced Topics in Data Engineering 2" at the Polytechnic University of Barcelona (UPC), thanks to the initiative that was taken to ensure that students in the Data Engineering degree have knowledge of the software engineering best practices that must be followed to build quality work within the ML industry.

Summary of the project
------------

Below, we provide two links for quick access to the dataset card and model card that summarize our project.

- [Dataset card](https://github.com/MLOps-essi-upc/taed2-PedestrianDetection/blob/main/data/dataset-card.md)

- [Model card](https://github.com/MLOps-essi-upc/taed2-PedestrianDetection/blob/main/models/model-card.md)

Project Organization
------------

    ├── LICENSE
    ├── README.md                       <- The top-level README for developers using this project
    │
    ├── .dvc
    │   ├── .gitignore
    │   └── config
    ├── .coverage
    ├── .dvcignore
    ├── .gitignore
    │
    ├── data
    │   ├── .gitignore
    │   ├── processed                   <- The final, canonical data sets for modeling  (MIRAR)
    │   ├── raw                         <- The original, immutable data dump
    │       └── .gitignore
    │   └── dataset-card.md             <- Dataset card containing dataset information
    │
    ├── dvc.lock 
    ├── dvc.yaml
    │
    ├── gx                              <- Great Expectations configuration and tests folder
    │   ├── .gitignore
    │   ├── checkpoints
    │   │   └── my_checkpoint.yml
    │   ├── expectations
    │   │   ├── .ge_store_backend_id
    │   │   └── pennfundan-training_suite.json
    │   ├── plugins/custom_data_docs/styles
    │   │   └── data_docs_custom_styles.css
    │   ├── uncommitted
    │   │   ├── data_docs/local_site
    │   │   │   └── index.html          <- Great Expectations results
    │   │   ├── validation
    │   │   └── config_variables.yml
    │   └── great_expectations.yml
    │
    ├── metrics                         <- Metrics folder
    │   └── emissions.csv
    │
    ├── models                          <- Trained and serialized models
    │   ├── .gitignore
    │   ├── Structure of model Mask R-CNN.png
    │   ├── baseline.pth.dvc
    │   ├── model-card.md               <- Model card containing model information
    │   └── other_trained_models.dvc
    │
    ├── notebooks                       <- Jupyter notebooks for exploration
    │   ├── .gitignore
    │   ├── 1.0-rb-data_processing.ipynb
    │   └── 1.0-cmp-modelling_pipeline.ipynb
    │
    ├── params.yaml  
    ├── pyproject.toml  
    │
    ├── requirements.txt                <- The requirements file for reproducing the analysis 
    │                                   environment, generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                        <- makes project pip installable so src can be imported
    │
    ├── src                             <- Source code for use in this project
    │   ├── .gitignore                       
    │   ├── app                         <- FastAPI app directory
    │   │   ├── api.py
    │   │   ├── draw_mask_map.py
    │   │   └── draw_segmentation_map.py
    │   │
    │   ├── data                        <- Scripts to download or generate data
    │   │   ├── .gitignore
    │   │   ├── coco_eval.py
    │   │   ├── coco_utils.py
    │   │   ├── data_downloading.py
    │   │   ├── data_preprocessing.py
    │   │   ├── engine.py
    │   │   ├── pedestrian_dataset_class.py
    │   │   ├── transforms.py
    │   │   ├── utils.py
    │   │   └── validate.py
    │   │
    │   ├── features                    <- Scripts to turn raw data into features for modeling
    │   │   ├── .gitignore
    │   │   ├── 
    │   │   └── 
    │   │
    │   └── models                      <- Scripts to train models
    │       ├── .gitingore
    │       ├── data.py
    │       ├── modelling.py
    │       ├── modelling_pipeline.py
    │       └── testing_model.py
    │
    ├── tests                           <- PyTest testing scripts
    │   ├── .gitignore
    │   ├── out
    │   │   └── tests-report.xml
    │   ├── __init__.py
    │   ├── test_api.py
    │   ├── test_models.py
    │   ├── test_negative_det.py
    │   └── test_positive_det.py
    │
    ├── test_environtment.py   
    └── tox.ini                         <- tox file with settings for running tox


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
