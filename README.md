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
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
