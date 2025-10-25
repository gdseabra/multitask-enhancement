from typing import Any, Optional, Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from data.components.enhancer_predict_dataset import EnhancerPredictionDataset


class EnhancerPredictionDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        img_subdir: str = "/latents/",
        data_list: str = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_predict: Optional[Dataset] = None


        self.batch_size_per_device = batch_size

        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.skel_transforms = transforms.Compose(
            [transforms.ToTensor()]
        )


    def setup(self, stage: Optional[str] = None):
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_predict:
            self.dataset = EnhancerPredictionDataset(self.hparams.data_dir, 
                                                  transform=self.transforms, 
                                                  data_list=self.hparams.data_list, 
                                                  img_subdir=self.hparams.img_subdir, 
                                                  )



    def pred_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )