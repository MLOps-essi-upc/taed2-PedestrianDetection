import torch
from torch.optim.lr_scheduler import StepLR
from engine import train_one_epoch, evaluate  # imported from vision.git, it ha st be cloned!
import mlflow
import mlflow.pytorch
from mlflow import log_metric, log_param, log_params, log_artifacts


def evaluation_mflow(name_run, data_loader_val, model, device):
    model.eval()

    # Start an MLflow run
    mlflow.start_run(run_name=name_run)

    evaluator = evaluate(model, data_loader_val, device=device)

    # Log the metrics
    mlflow.log_metrics({
        "AP_bb": evaluator.coco_eval["bbox"].stats[0],
        "AR_bb": evaluator.coco_eval["bbox"].stats[8],
        "AP_segm": evaluator.coco_eval["segm"].stats[0],
        "AR_segm": evaluator.coco_eval["segm"].stats[8],
    })

    # End the MLflow run
    mlflow.end_run()


def train_mlflow(model, data_loader, data_loader_val, num_epochs, hidden_layer, batch_size_train, name, device):

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

    # Log the trained model
    mlflow.pytorch.log_model(model, name)

    mlflow.end_run()
