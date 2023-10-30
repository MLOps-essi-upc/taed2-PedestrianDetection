"""
Setup script for the Instance Segmentation for Pedestrian Detection project.

This script is used to configure and install the 'src' package.
The project focuses on instance segmentation for pedestrian detection.

Authors: Rodrigo Bonferroni, Arlet Corominas and Clàudia Mur
License: MIT
"""

from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description="""This project is part of the Advanced 
    Data Science Topics II course at Universitat Politècnica 
    de Catalunya (UPC). It addresses instance segmentation 
    for pedestrian detection using deep learning.""",
    author='Rodrigo Bonferroni, Arlet Corominas & Clàudia Mur',
    license='MIT',
)
