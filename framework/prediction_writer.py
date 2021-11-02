import os
import numpy as np
from typing import List, Any
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.core.lightning import LightningModule
from . import Datasets


class PredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        write_interval: str = "epoch",
        out_file_name: str = None,
    ):
        super().__init__(write_interval)
        self.file_path = out_file_name
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, mode="w") as writer:
            writer.truncate()

    def write_on_batch_end(
        self,
        trainer,
        pl_module: "LightningModule",
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        with open(self.file_path, mode="a") as writer:
            np.savetxt(writer, prediction.detach().cpu().int().numpy(), fmt="%d")

    def write_on_epoch_end(
        self,
        trainer,
        pl_module: "LightningModule",
        predictions: List[Any],
        batch_indices: List[Any],
    ):
        with open(self.file_path, mode="a") as writer:
            np.savetxt(writer, predictions.detach().cpu().int().numpy(), fmt="%d")
