from typing import Any, Optional, Tuple

import lightning as L
import torch
import torch.nn.functional as F  
from torch.utils.data import (
    DataLoader,
    Dataset,
    default_collate,  
    random_split,
)
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

    # ---------------------------------------------------------------------
    # vvv ADDED FUNCTION vvv
    # ---------------------------------------------------------------------
    @staticmethod
    def custom_collate_fn(batch: Any) -> Any:
        """
        Custom collate function to pad tensors to the same size.
        
        Assumes the tensor that needs padding is the *first* element 
        in the sample tuple, and its shape is [C, H, W].
        """
        # Check if the batch is empty
        if not batch:
            return batch
        
        elem = batch[0]
        
        # Case 1: __getitem__ returns a tuple (e.g., (image, other_data))
        if isinstance(elem, (tuple, list)):
            # Transpose the batch:
            # from: [(img1, data1), (img2, data2), ...]
            # to:   ([img1, img2, ...], [data1, data2, ...])
            transposed_batch = list(zip(*batch))
            
            # Get the tensors that need padding (assuming they are the first element)
            images = transposed_batch[0]
            
            # Find max dimensions (assuming [C, H, W])
            # Your error showed H (dim 1) was variable: [1, 750, 800] vs [1, 752, 800]
            max_h = max(img.shape[1] for img in images)
            max_w = max(img.shape[2] for img in images)

            padded_images = []
            for img in images:
                # Calculate padding (bottom, right)
                pad_h = max_h - img.shape[1]
                pad_w = max_w - img.shape[2]
                
                # F.pad format: (pad_left, pad_right, pad_top, pad_bottom)
                # We pad the last dim (W) first, then the second-to-last (H)
                padding = (0, pad_w, 0, pad_h)
                padded_images.append(F.pad(img, padding, "constant", 0))
            
            # Stack the newly padded images
            images_batch = torch.stack(padded_images)
            
            # Collate the rest of the data (if any) using the default collate
            other_data_batches = [default_collate(part) for part in transposed_batch[1:]]
            
            # Return the collated batch
            return (images_batch, *other_data_batches)
        
        # Case 2: __getitem__ returns just the tensor
        elif isinstance(elem, torch.Tensor):
            images = batch
            max_h = max(img.shape[1] for img in images)
            max_w = max(img.shape[2] for img in images)

            padded_images = []
            for img in images:
                pad_h = max_h - img.shape[1]
                pad_w = max_w - img.shape[2]
                padding = (0, pad_w, 0, pad_h)
                padded_images.append(F.pad(img, padding, "constant", 0))

            return torch.stack(padded_images)

        # Fallback for other data types (e.g., dicts)
        else:
            return default_collate(batch)

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
            collate_fn=self.custom_collate_fn  # <-- MODIFIED LINE
        )