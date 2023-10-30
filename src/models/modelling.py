"""
Module Name: modelling.py

This module provides utility functions for training and evaluating models with MLflow integration.
"""
import os
import sys


import mlflow
import mlflow.pytorch
import pandas as pd
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'*2))
sys.path.insert(1, os.path.join(root_dir, 'src/data'))

from engine import train_one_epoch, evaluate


def evaluation_mflow(name_run, data_loader_val, model, device, emissions_file = "emissions.csv"):
    """
    Perform evaluation of a model using MLflow.

    Args:
    name_run (str): The name of the MLflow run.
    data_loader_val (DataLoader): The validation data loader.
    model (nn.Module): The model to be evaluated.
    device (torch.device): The device to run the evaluation on.

    Returns:
    None
    """
    model.eval()

    # Start an MLflow run
    mlflow.start_run(run_name=name_run)

    evaluator = evaluate(model, data_loader_val, device=device)


    # Log the metrics
    mlflow.log_metrics({
        "eval_AP_bb": evaluator.coco_eval["bbox"].stats[0],
        "eval_AR_bb": evaluator.coco_eval["bbox"].stats[8],
        "eval_AP_segm": evaluator.coco_eval["segm"].stats[0],
        "eval_AR_segm": evaluator.coco_eval["segm"].stats[8],
    })

    # Log the CO2 emissions to MLflow
    emissions = pd.read_csv(emissions_file)
    emissions_metrics = emissions.iloc[-1, 4:13].to_dict()
    emissions_params = emissions.iloc[-1, 13:].to_dict()
    mlflow.log_params(emissions_params)
    mlflow.log_metrics(emissions_metrics)

    # End the MLflow run
    mlflow.end_run()


def train_mlflow(model, data_loader, data_loader_val, num_epochs, hidden_layer,
                 batch_size_train, name, device, emissions_file="emissions.csv"):
    """
    Train a model and log training metrics and parameters to MLflow.

    Args:
    model (nn.Module): The model to be trained.
    data_loader (DataLoader): The training data loader.
    data_loader_val (DataLoader): The validation data loader.
    num_epochs (int): The number of training epochs.
    hidden_layer (int): The size of the hidden layer for the mask classifier.
    batch_size_train (int): The batch size for training.
    name (str): The name of the MLflow run.
    device (torch.device): The device to run the training on.

    Returns:
    None
    """

    if mlflow.active_run():
        mlflow.end_run()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    mlflow.start_run(run_name=name)

    # Log parameters
    mlflow.log_params({
        "hidden_size_mask_classifier": hidden_layer,
        "batch_size_train": batch_size_train,
        "num_epochs": num_epochs,

    })

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # Log train metrics
        mlflow.log_metrics({
            "loss": metric_logger.meters['loss'].global_avg,
            "loss_classifier": metric_logger.meters['loss_classifier'].global_avg,
            "loss_box_reg": metric_logger.meters['loss_box_reg'].global_avg,
            "loss_mask": metric_logger.meters['loss_mask'].global_avg,
            "loss_objectness": metric_logger.meters['loss_objectness'].global_avg,
            "loss_rpn_box_reg": metric_logger.meters['loss_rpn_box_reg'].global_avg,
        }, step=epoch)
        evaluator = evaluate(model, data_loader, device=device)
        mlflow.log_metrics({
            "train_AP_bb": evaluator.coco_eval["bbox"].stats[0],
            "train_AR_bb": evaluator.coco_eval["bbox"].stats[8],
            "train_AP_segm": evaluator.coco_eval["segm"].stats[0],
            "train_AR_segm": evaluator.coco_eval["segm"].stats[8],
        }, step=epoch)

        # Log eval metrics
        evaluator = evaluate(model, data_loader_val, device=device)
        mlflow.log_metrics({
            "eval_AP_bb": evaluator.coco_eval["bbox"].stats[0],
            "eval_AR_bb": evaluator.coco_eval["bbox"].stats[8],
            "eval_AP_segm": evaluator.coco_eval["segm"].stats[0],
            "eval_AR_segm": evaluator.coco_eval["segm"].stats[8],
        }, step=epoch)

        model.train()

        # update the learning rate
        lr_scheduler.step()

    # Log the CO2 emissions to MLflow
    emissions = pd.read_csv(emissions_file)
    emissions_metrics = emissions.iloc[-1, 4:13].to_dict()
    emissions_params = emissions.iloc[-1, 13:].to_dict()
    mlflow.log_params(emissions_params)
    mlflow.log_metrics(emissions_metrics)

    # Log the trained model
    mlflow.pytorch.log_model(model, name)

    mlflow.end_run()
