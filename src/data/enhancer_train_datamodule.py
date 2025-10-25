from typing import Any, Optional, Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from data.components.enhancer_train_dataset import EnhancerTrainDataset #, PatchEnhancerTrainDataset


class EnhancerTrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        lat_subdir: str = "/latents/",
        ref_subdir: str = '/references/',
        skel_subdir: str = '/skel/',
        mask_subdir: str = '/masks/',
        occ_mask_subdir: str = '/occ_masks/',
        bin_subdir: str = '/bin/',
        mnt_subdir: str = '/mnt/',
        orient_subdir: str = '/orient/', 
        apply_mask: bool = True,
        use_ref_mask: bool = False,
        data_list: str = None,
        batch_size: int = 64,
        train_val_split: Tuple[float, float] = (0.7, 0.3),
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


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

        if not self.data_train and not self.data_val:
            dataset = EnhancerTrainDataset(self.hparams.data_dir, 
                                                  transform=self.transforms, 
                                                  skel_transform=self.skel_transforms,
                                                  data_list=self.hparams.data_list, 
                                                  lat_subdir=self.hparams.lat_subdir, 
                                                  ref_subdir=self.hparams.ref_subdir, 
                                                  skel_subdir=self.hparams.skel_subdir,
                                                  bin_subdir=self.hparams.bin_subdir,
                                                  mask_subdir=self.hparams.mask_subdir,
                                                  mnt_subdir=self.hparams.mnt_subdir,
                                                  orient_subdir=self.hparams.orient_subdir,
                                                  apply_mask =self.hparams.apply_mask,
                                                  use_ref_mask=self.hparams.use_ref_mask
                                                  )

            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
