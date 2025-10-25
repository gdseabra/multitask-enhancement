from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
import numpy as np
import os
from PIL import Image
# from torchmetrics.classification.accuracy import Accuracy

import torch.nn.functional as F

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    # sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def bce_loss(pred, target, mask_label, mnt_label):
    bce_criterion = nn.functional.l1_loss
    image_loss = bce_criterion(pred, target, reduction = 'none')

    minutia_weighted_map = mnt_label
    image_loss *= minutia_weighted_map
    # image_loss = image_loss * mask_label
    # return torch.sum(image_loss) / (torch.sum(mask_label).clamp(min=1) + 1e-7)

    return torch.mean(torch.sum(image_loss, dim=(1,2)))

class MyCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, pixel_weight, mask):
        enh_loss = bce_loss(input, target, pixel_weight, mask)

        return enh_loss

class MyWeightedL1Loss(nn.L1Loss):
    def __init__(self, reduction='none'):
        super(MyWeightedL1Loss, self).__init__(reduction=reduction)

    def forward(self, input, target):
        pixel_mae = super(MyWeightedL1Loss, self).forward(input, target)
        loss = pixel_mae
        return loss.sum()/(loss.size(0)) # mean per-image loss (not per-pixel or per-batch).

class MaskedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask):
        """
        :param logits: Tensor of shape (N, 1, H, W) - raw model outputs
        :param targets: Tensor of shape (N, 1, H, W) - binary labels
        :param mask: Tensor of shape (N, 1, H, W) - binary mask (1=foreground, 0=background)
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        foreground = mask.bool()

        # Avoid empty masks by clamping denominator
        foreground_loss = bce_loss[foreground].mean() if foreground.any() else torch.tensor(0.0, device=logits.device)

        return foreground_loss

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask):
        """
        :param logits: Tensor of shape (N, 1, H, W) - raw model outputs
        :param targets: Tensor of shape (N, 1, H, W) - binary labels
        :param mask: Tensor of shape (N, 1, H, W) - binary mask (1=foreground, 0=background)
        """
        mse_loss = F.mse_loss(logits, targets, reduction='none')

        foreground = mask.bool()

        # Avoid empty masks by clamping denominator
        foreground_loss = mse_loss[foreground].mean() if foreground.any() else torch.tensor(0.0, device=logits.device)

        return foreground_loss

def circular_smooth_labels(true_idx, num_classes=90, sigma=1.5):
    """
    true_idx: (B, H, W) integer tensor of ground truth classes
    returns: (B, num_classes, H, W) smoothed probability tensor
    """
    device = true_idx.device
    B, H, W = true_idx.shape
    classes = torch.arange(num_classes, device=device).view(1, num_classes, 1, 1)

    # Expand true_idx to match shape
    true_idx_exp = true_idx.unsqueeze(1)

    # Circular distance between each class and ground truth
    diff = torch.abs(classes - true_idx_exp)
    diff = torch.minimum(diff, num_classes - diff)  # wrap-around distance

    # Gaussian weighting
    weights = torch.exp(-0.5 * (diff.float() / sigma) ** 2)
    weights = weights / weights.sum(dim=1, keepdim=True)

    return weights


def circular_cross_entropy(pred_logits, true_idx, num_classes=90, sigma=1.5):
    # pred_logits: (B, num_classes, H, W)
    # true_idx: (B, H, W)
    # soft_targets = circular_smooth_labels(true_idx, num_classes, sigma)  # (B, C, H, W)
    # Initialize cross-entropy loss (no label smoothing here, we provide soft labels directly)

    # Initialize class weights: class 90 has 10x higher weight
    # class_w = torch.ones(num_classes, device=pred_logits.device, dtype=pred_logits.dtype)
    # class_w[90] = 1.0

    # Cross-entropy with class weighting
    criterion = nn.CrossEntropyLoss(ignore_index=90)

    # criterion expects soft_targets float tensor of shape (B, C, H, W)
    # ⚠️ NOTE: As of PyTorch ≥ 1.10, CrossEntropyLoss accepts soft labels directly.
    # loss = criterion(pred_logits, soft_targets)
    loss = criterion(pred_logits, true_idx)
    return loss



class EnhancerLitModule(LightningModule):
    """
    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        output_path: str = None,
        patch_size: Tuple[int, int] = (128, 128),
        use_patches: bool = False,
        stride: int = 8,
        warmup_epochs_dirmap: int = 2

    ) -> None:
        """Initialize a `EnhancerLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = nn.BCEWithLogitsLoss()

        self.mse_criterion = torch.nn.functional.mse_loss
        self.bce_criterion = torch.nn.functional.binary_cross_entropy_with_logits

        self.ce_criterion = torch.nn.functional.cross_entropy


        self.patch_size = patch_size
        self.use_patches = use_patches
        self.stride = stride
        self.input_row = self.patch_size[0]
        self.input_col = self.patch_size[1]
        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        # self.val_acc = Accuracy(task="multiclass", num_classes=10)
        # self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        self.output_path = output_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        # self.val_acc.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y_dirmap, y_orig, y_bin = batch
        pred_enh = self.forward(x)

        pred_orig  = pred_enh[:,0,:,:]
        pred_bin = pred_enh[:,1,:,:]


        true_orig   = y_orig[:, 0, :, :]
        true_bin    = y_bin[:, 0, :, :]

        # true_seg = 1 - y[:, 90, :, :]  # -> (B, H, W)
        # pred_seg = pred_seg.squeeze(1)  # (B, 1, H, W) -> (B, H, W)


        enh_loss = (0.5 * self.mse_criterion(pred_orig, true_orig) + 0.5 * self.bce_criterion(pred_bin, true_bin))

        # total_loss = self.bce_criterion(pred_dirmap, true_dirmap)

        total_loss = enh_loss

        # loss = (self.criterion(pred_bin, true_bin) + dice_loss(F.sigmoid(pred_bin), true_bin, multiclass=False))
        # loss += 0.5 * self.mse_criterion(pred_orig, true_orig)
        # loss = self.mse_criterion(yhat, y_skel,  torch.ones_like(y_skel))


        # for i, name in enumerate(names):
        #     mnt = mnt_map[i, :, :]

        #     mnt = mnt.cpu().numpy()


        #     mnt = (255 * (mnt - np.min(mnt))/(np.max(mnt) - np.min(mnt))).astype('uint8')

        #     mnt = Image.fromarray(mnt)
        #     # print(name)
        #     mnt.save(self.output_path + '/mnt/' + str(i) + '.png')
        
        return total_loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # return loss or backpropagation will fail
        return loss
    
    def on_train_epoch_start(self):
        """
        Hook executado no início de cada época de treinamento.
        Ideal para congelar/descongelar camadas.
        """
        pass

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        # self.test_acc(preds, targets)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def save_orientation_field(self, output_tensor: torch.Tensor, mask:np.ndarray, png_path: str, dir_path: str):
        """
        Save the orientation field from a 90-channel output tensor to a .png image and .dir text file.

        Parameters
        ----------
        output_tensor : torch.Tensor
            Tensor of shape [1, 90, H, W] with probabilities or responses per angle.
        png_path : str
            Path to save the .png image.
        dir_path : str
            Path to save the .dir text file.
        """
        # Ensure tensor is detached and on CPU
        output = output_tensor.cpu().squeeze(0)  # -> [90, H, W]
        assert output.shape[0] == 90, "Output must have 91 channels"

        # Find channel with maximum response per pixel
        max_indices = torch.argmax(output, dim=0).numpy()  # -> [H, W]

        # Convert channel indices to angles (0°, 2°, ..., 178°)
        angles = max_indices * 2  # [H, W] int

        # Map 180° to -1 for background
        background = -np.ones_like(angles)

        if mask != None:
            angles = np.where(mask == 0, background, angles)

        H, W = angles.shape

        # --- Save PNG using PIL ---
        # Scale angles (0..178) → (0..255) for visualization
        img_array = (angles.astype(np.float32) * (255.0 / 178.0)).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
        img.save(png_path)

        # --- Save .dir file with multiple columns per line ---
        with open(dir_path, "w") as f:
            f.write(f"{W} {H}\n")  # width and height
            for y in range(H):
                row_values = " ".join(str(angles[y, x]) for x in range(W))
                f.write(row_values + "\n")

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        data  = batch[0]
        names = batch[1]
        x, y = batch

        # print(data.shape) # (28,1,128,128)
        # print(y) # tupla com todos os nomes das imagens do batch

        gabor_path = os.path.join(self.output_path, "gabor")
        if not os.path.exists(gabor_path):
            os.makedirs(gabor_path)

        bin_path = os.path.join(self.output_path, "bin")
        if not os.path.exists(bin_path):
            os.makedirs(bin_path)

        enh_path = os.path.join(self.output_path, "enh")
        if not os.path.exists(enh_path):
            os.makedirs(enh_path)

        dirmap_path = os.path.join(self.output_path, "dirmap")
        if not os.path.exists(dirmap_path):
            os.makedirs(dirmap_path)

        dirmap_png_path = os.path.join(self.output_path, "dirmap_png")
        if not os.path.exists(dirmap_png_path):
            os.makedirs(dirmap_png_path)

        seg_path = os.path.join(self.output_path, "mask")
        if not os.path.exists(seg_path):
            os.makedirs(seg_path)

        if not self.use_patches:
            # dirmap_pred, latent_enh = self.forward(x)
            latent_enh = self.forward(x)
        else:
            shape_latent = data.shape
            ROW = shape_latent[2]
            COL = shape_latent[3]
            row_list_1 = range(self.input_row, ROW+1, self.stride)
            row_list_2 = range(ROW, row_list_1[-1]-1,-self.stride)
            row_list = [*row_list_1, *row_list_2]
            
            col_list_1 = range(self.input_col, COL+1, self.stride)
            col_list_2 = range(COL, col_list_1[-1]-1, -self.stride)
            col_list = [*col_list_1,*col_list_2]

            patch_ind = 0

            latent_enh = torch.zeros((data.shape[0], 2, data.shape[2], data.shape[3]), device=x.device)
            
            for row_ind in row_list:
                for col_ind in col_list:
                    patch_pred = self.forward(data[:,:,(row_ind-self.input_row):row_ind,(col_ind-self.input_col):col_ind])
                    latent_enh[:,:,(row_ind-self.input_row):row_ind, (col_ind-self.input_col):col_ind] += patch_pred

        for i, name in enumerate(names):
            gabor   = latent_enh[i, 1, :, :]
            orig    = latent_enh[i, 0, :, :]

            gabor   = torch.nn.functional.sigmoid(gabor)
            bin   = torch.round(gabor)

            gabor = gabor.cpu().numpy()
            bin   = bin.cpu().numpy()
            orig  = orig.cpu().numpy()

            gabor = (255 * (gabor - np.min(gabor))/(np.max(gabor) - np.min(gabor))).astype('uint8')
            bin   = (255 * (bin - np.min(bin))/(np.max(bin) - np.min(bin))).astype('uint8')
            orig   = (255 * (orig - np.min(orig))/(np.max(orig) - np.min(orig))).astype('uint8')

            gabor = Image.fromarray(gabor)
            gabor.save(gabor_path + '/' + name + '.png')

            bin = Image.fromarray(bin)
            bin.save(bin_path + '/' + name + '.png')

            orig = Image.fromarray(orig)
            orig.save(enh_path + '/' + name + '.png')

            # dirmap   = dirmap_pred[i, :, :, :]
            # dirmap   = torch.nn.functional.sigmoid(dirmap)

            
            # self.save_orientation_field(dirmap, None, f"{dirmap_png_path}/{name}.png", f"{dirmap_path}/{name}.dir")


            # bin   = torch.nn.functional.sigmoid(gabor)
            # bin   = torch.round(bin)

            # gabor = gabor.cpu().numpy()
            # bin   = bin.cpu().numpy()
            # orig  = orig.cpu().numpy()

            # gabor = (255 * (gabor - np.min(gabor))/(np.max(gabor) - np.min(gabor))).astype('uint8')
            # bin   = (255 * (bin - np.min(bin))/(np.max(bin) - np.min(bin))).astype('uint8')

            # mask = (255 * (mask - np.min(mask))/(np.max(mask) - np.min(mask))).astype('uint8')
            # mask_img = Image.fromarray(mask)
            # mask_img.save(seg_path + '/' + name + '.png')




if __name__ == "__main__":
    _ = EnhancerLitModule(None, None, None, None)
