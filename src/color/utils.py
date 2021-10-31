import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import (accuracy, confusion_matrix, f1, precision,
                                     recall)

from color import (ColorDataModule, ColorDatasets, ColorModel,
                   ColorPredictionWriter)


def numel(m: nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = m.parameters()
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def get_conf_data(train_dataset: ColorDatasets, test_dataset: ColorDatasets, model_arch):
    prediction_root = os.path.join("predictions", 'color')
    predict_model_name = f"{train_dataset.name}_{model_arch}"
    best_model_path = os.path.join("checkpoints", "color", f"best_{predict_model_name}.ckpt")
    prediction_out_file = f"{test_dataset.name}_by_{predict_model_name}.txt"
    return prediction_root, predict_model_name, best_model_path, prediction_out_file


def predict_and_persist_color(
    prediction_root,
    predict_model_name,
    best_model_path,
    prediction_out_file,
    test_dataset:ColorDatasets,
    batch_size,
    allowed_color_list,
    gpus=None,
    num_dataloader_workers=1,
):
    # Callback to persist prediction
    predict_callback = ColorPredictionWriter(
        prediction_root,
        write_interval="batch",
        dataset_name=test_dataset.name,
        out_file_name=prediction_out_file,
    )
    # datamodule
    dm = ColorDataModule(
        dataset_type=test_dataset,
        data_dir=f"dataset/",
        allowed_color_list=allowed_color_list,
        with_predictions=True,
        batch_size=batch_size,
        num_workers = num_dataloader_workers
        #  prediction_file='train_label.xml'
    )
    # init model from checkpoint
    model = ColorModel.load_from_checkpoint(best_model_path)
    # init trainer
    trainer = pl.Trainer(
        gpus=gpus, progress_bar_refresh_rate=20, callbacks=[predict_callback]
    )
    dm.setup("test")
    dl = DataLoader(dm.test_dataset, batch_size=batch_size)
    trainer.predict(model, dataloaders=dl)


def evaluate_predictions(
    prediction_root,
    predict_model_name,
    best_model_path,
    prediction_out_file,
    test_dataset: ColorDatasets,
    allowed_color_list,
):

    prediction_out_file_path = f"{prediction_root}/{prediction_out_file}"
    # load dataset with ground truth
    dm_gt = ColorDataModule(
        dataset_type=test_dataset,
        data_dir=f"dataset/",
        allowed_color_list=allowed_color_list,
        with_predictions=True,
        #  prediction_file='train_label.xml'
    )
    dm_gt.setup("test")
    # gt will be indexed based on the dataset
    print(f"{test_dataset.name} using {predict_model_name}")
    colors = dm_gt.test_dataset.colors
    gt = np.array(colors)
    gt = torch.from_numpy(gt)
    predictions = np.loadtxt(prediction_out_file_path, dtype=int)
    predictions = torch.from_numpy(predictions)
    avg_met = "weighted"
    num_classes = len(allowed_color_list)
    accuracy_val = accuracy(predictions, gt, average=avg_met, num_classes=num_classes)
    precision_val = precision(predictions, gt, average=avg_met, num_classes=num_classes)
    f1_val = f1(predictions, gt, average=avg_met, num_classes=num_classes)
    recall_val = recall(predictions, gt, average=avg_met, num_classes=num_classes)
    print(f"The accuracy is {accuracy_val}")
    print(f"The precision is {precision_val}")
    print(f"The f1 is {f1_val}")
    print(f"The recall is {recall_val}")

    # print confusion matrix
    cm = confusion_matrix(predictions, gt, num_classes, "true")
    cm = ConfusionMatrixDisplay(cm.numpy(), display_labels=allowed_color_list)
    cm.plot(
        cmap=plt.cm.Blues,
        values_format=".2f",
    )
    plt.show()

    return accuracy_val, precision_val, f1_val, recall_val
