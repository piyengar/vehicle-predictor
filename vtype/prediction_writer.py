import os
from . import TypeDatasets
import numpy as np
from typing import List, Any
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.core.lightning import LightningModule


class TypePredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        write_interval: str = "epoch",
        dataset_type: TypeDatasets = TypeDatasets.VERI,
        out_file_name: str = None,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.dataset_type = dataset_type
        self.file_name = (
            out_file_name if out_file_name is not None else f"{self.dataset_type.name}.txt"
        )
        with open(os.path.join(self.output_dir, self.file_name), mode="w") as writer:
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
        with open(os.path.join(self.output_dir, self.file_name), mode="a") as writer:
            np.savetxt(writer, prediction.detach().cpu().int().numpy(), fmt="%d")

    def write_on_epoch_end(
        self,
        trainer,
        pl_module: "LightningModule",
        predictions: List[Any],
        batch_indices: List[Any],
    ):
        with open(os.path.join(self.output_dir, self.file_name), mode="a") as writer:
            np.savetxt(writer, predictions.detach().cpu().int().numpy(), fmt="%d")
