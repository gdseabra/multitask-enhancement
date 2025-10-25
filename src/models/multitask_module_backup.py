from typing import Any, Dict, Tuple, Callable

import torch
import torch.nn as nn
from torch import Tensor
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
import numpy as np
import os
from PIL import Image
import torch.nn.functional as F


class WeightedOrientationLoss(nn.Module):
    """
    Implements the weighted cross-entropy-like loss from the paper.
    
    This loss encourages the correct orientation channel to have a high probability
    while penalizing high probabilities in incorrect channels.
    
    L_* = -1/|ROI| * sum_{ROI} sum_{i=1 to N} (lambda+ * p_l * log(p) + 
                                               lambda- * (1-p_l) * log(1-p))
    """
    def __init__(self, lambda_pos: float = 1.0, lambda_neg: float = 1.0, epsilon: float = 1e-8):
        """
        Args:
            lambda_pos (float): Weight for the positive samples (correct orientation).
            lambda_neg (float): Weight for the negative samples (incorrect orientations).
            epsilon (float): A small value to ensure numerical stability in log operations.
        """
        super().__init__()
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction (torch.Tensor): The model's output probability map.
                                       Expected shape: (B, N, H, W), where N is the number of orientation bins.
                                       Values should be probabilities (e.g., after a sigmoid).
            target (torch.Tensor): The ground truth orientation map with integer labels.
                                   Expected shape: (B, 1, H, W).
            mask (torch.Tensor): The Region of Interest (ROI) mask. Loss is only computed where mask is 1.
                                 Expected shape: (B, 1, H, W).
                                 
        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        num_classes = prediction.shape[1]
        
        # Create one-hot encoded target from integer labels
        # Shape: (B, 1, H, W) -> (B, H, W, N) -> (B, N, H, W)
        target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Ensure mask is broadcastable
        # Shape: (B, 1, H, W)
        mask = mask.float()

        # Calculate the log probabilities for both terms
        log_p = torch.log(prediction + self.epsilon)
        log_one_minus_p = torch.log(1 - prediction + self.epsilon)

        # Calculate the positive and negative loss components
        # (lambda+ * p_l * log(p))
        loss_pos = self.lambda_pos * target_one_hot * log_p
        # (lambda- * (1-p_l) * log(1-p))
        loss_neg = self.lambda_neg * (1 - target_one_hot) * log_one_minus_p
        
        # Combine the two loss components
        loss = -(loss_pos + loss_neg)

        # Apply the ROI mask and calculate the mean loss per pixel in the ROI
        masked_loss = loss * mask
        
        # Normalize by the number of pixels in the ROI
        num_roi_pixels = mask.sum()
        if num_roi_pixels > 0:
            total_loss = masked_loss.sum() / num_roi_pixels
        else:
            # Avoid division by zero if mask is empty
            total_loss = torch.tensor(0.0, device=prediction.device)

        return total_loss

# ----------------------------------------------------------------------------

class OrientationCoherenceLoss(nn.Module):
    """
    Implements the orientation coherence loss (L_odpi) from the paper.
    
    This loss enforces that orientation predictions in a local neighborhood
    should be consistent or "coherent".
    
    L_odpi = |ROI| / (sum_{ROI} Coh) - 1
    
    where Coh is the coherence map calculated from the predicted orientation vectors.
    """
    def __init__(self, N: int = 90, epsilon: float = 1e-8):
        """
        Args:
            N (int): The number of discrete orientation angles.
            epsilon (float): A small value for numerical stability.
        """
        super().__init__()
        self.N = N
        self.epsilon = epsilon

        # Pre-compute angles and the 3x3 averaging kernel (J_3)
        # These will be registered as buffers and moved to the correct device
        # automatically with the module.
        
        # angle_i = floor(180/N) * i
        angle_step_deg = 180.0 / N
        angles_deg = torch.arange(N, dtype=torch.float32) * angle_step_deg
        
        # The formula uses cos(2 * angle) and sin(2 * angle)
        angles_rad_doubled = torch.deg2rad(2 * angles_deg)

        # Reshape for broadcasting with the prediction tensor (B, N, H, W)
        self.cos_terms = nn.Parameter(torch.cos(angles_rad_doubled).view(1, N, 1, 1), requires_grad=False)
        self.sin_terms = nn.Parameter(torch.sin(angles_rad_doubled).view(1, N, 1, 1), requires_grad=False)
        
        # J_3 is an all-ones 3x3 matrix for convolution (averaging)
        self.j3_kernel = nn.Parameter(torch.ones((1, 1, 3, 3), dtype=torch.float32), requires_grad=False)

    def forward(self, prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction (torch.Tensor): The model's output probability map.
                                       Expected shape: (B, N, H, W).
            mask (torch.Tensor): The Region of Interest (ROI) mask.
                                 Expected shape: (B, 1, H, W).
                                 
        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        # --- 1. Compute the averaging ridge orientation vector d_bar ---
        # d_bar_cos = (1/N) * sum(p_ori(i) * cos(2 * angle_i))
        d_bar_cos = torch.sum(prediction * self.cos_terms, dim=1, keepdim=True) / self.N
        # d_bar_sin = (1/N) * sum(p_ori(i) * sin(2 * angle_i))
        d_bar_sin = torch.sum(prediction * self.sin_terms, dim=1, keepdim=True) / self.N
        
        # --- 2. Calculate Coh = (d_bar * J_3) / (|d_bar| * J_3) ---
        
        # Numerator: Convolve vector components and find magnitude of the result
        # This averages the orientation vectors in a 3x3 neighborhood
        summed_d_cos = F.conv2d(d_bar_cos, self.j3_kernel, padding='same')
        summed_d_sin = F.conv2d(d_bar_sin, self.j3_kernel, padding='same')
        numerator = torch.sqrt(summed_d_cos**2 + summed_d_sin**2 + self.epsilon)
        
        # Denominator: Find magnitude of vectors, then convolve (average) them
        d_bar_mag = torch.sqrt(d_bar_cos**2 + d_bar_sin**2 + self.epsilon)
        denominator = F.conv2d(d_bar_mag, self.j3_kernel, padding='same')
        
        coh = numerator / (denominator + self.epsilon)
        
        # --- 3. Compute the final loss L_odpi ---
        mask = mask.float()
        roi_size = torch.sum(mask)
        
        if roi_size > 0:
            # Sum coherence only over the ROI
            sum_coh_roi = torch.sum(coh * mask)
            loss = roi_size / (sum_coh_roi + self.epsilon) - 1.0
        else:
            loss = torch.tensor(0.0, device=prediction.device)
            
        return loss

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    fn = dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def bce_loss(pred, target, mask_label, mnt_label):
    bce_criterion = nn.functional.l1_loss
    image_loss = bce_criterion(pred, target, reduction = 'none')
    minutia_weighted_map = mnt_label
    image_loss *= minutia_weighted_map
    return torch.mean(torch.sum(image_loss, dim=(1,2)))

class MyCriterion(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input, target, pixel_weight, mask): return bce_loss(input, target, pixel_weight, mask)

class MyWeightedL1Loss(nn.L1Loss):
    def __init__(self, reduction='none'): super(MyWeightedL1Loss, self).__init__(reduction=reduction)
    def forward(self, input, target): return super(MyWeightedL1Loss, self).forward(input, target).sum()/(loss.size(0))

class MaskedBCELoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, logits, targets, mask):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        foreground = mask.bool()
        return bce_loss[foreground].mean() if foreground.any() else torch.tensor(0.0, device=logits.device)

class MaskedMSELoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, logits, targets, mask):
        mse_loss = F.mse_loss(logits, targets, reduction='none')
        foreground = mask.bool()
        return mse_loss[foreground].mean() if foreground.any() else torch.tensor(0.0, device=logits.device)

def circular_smooth_labels(true_idx, num_classes=90, sigma=1.5):
    device, (B, H, W) = true_idx.device, true_idx.shape
    classes = torch.arange(num_classes, device=device).view(1, num_classes, 1, 1)
    true_idx_exp = true_idx.unsqueeze(1)
    diff = torch.minimum(torch.abs(classes - true_idx_exp), num_classes - torch.abs(classes - true_idx_exp))
    weights = torch.exp(-0.5 * (diff.float() / sigma) ** 2)
    return weights / weights.sum(dim=1, keepdim=True)

def circular_cross_entropy(pred_logits, true_idx, num_classes=90, sigma=1.5):
    criterion = nn.CrossEntropyLoss(ignore_index=90)
    return criterion(pred_logits, true_idx)
# <-------------------------------------------------------------------------------------->


class EnhancerLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer_dirmap: Callable,
        optimizer_enh: Callable,
        scheduler_dirmap: Callable,
        scheduler_enh: Callable,
        compile: bool,
        output_path: str = None,
        patch_size: Tuple[int, int] = (128, 128),
        use_patches: bool = False,
        stride: int = 8,
        warmup_epochs_dirmap: int = 2
    ) -> None:
        super().__init__()
        # 🔻 REMOVIDO: LRs não são mais parâmetros diretos.
        # Elas estão dentro das configurações dos otimizadores.
        self.save_hyperparameters(logger=False, ignore=["net"]) # Ignora a rede para não salvar o grafo no checkpoint
        self.net = net

        # Esta linha informa ao Lightning para desativar seu     #
        # loop de otimização padrão.                              #
        self.automatic_optimization = False

        self.mse_criterion = torch.nn.functional.mse_loss
        self.bce_criterion = torch.nn.functional.binary_cross_entropy_with_logits
        self.train_loss = MeanMetric()
        self.train_ori_loss = MeanMetric()
        self.train_enh_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_ori_loss = MeanMetric()
        self.val_enh_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        self.patch_size = patch_size
        self.use_patches = use_patches
        self.stride = stride
        self.output_path = output_path


    # ... (forward, on_train_start, model_step, training_step, on_train_epoch_start,
    #      validation_step, etc. permanecem EXATAMENTE os mesmos) ...
    # <-------------------------------------------------------------------------------------->
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)
    
    def on_train_start(self) -> None:
        self.val_loss.reset(); self.val_ori_loss.reset(); self.val_enh_loss.reset(); self.val_loss_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, y_dirmap, y_orig, y_bin = batch
        pred_dirmap, pred_enh = self.forward(x)
        pred_orig, pred_bin = pred_enh[:,0,:,:], pred_enh[:,1,:,:]
        true_orig, true_bin = y_orig[:, 0, :, :], y_bin[:, 0, :, :]
        # true_dirmap = F.interpolate(y_dirmap, size=true_bin.shape[1:], mode="bilinear", align_corners=False)
        true_dirmap_idx = y_dirmap.argmax(dim=1)
        ori_loss = circular_cross_entropy(pred_dirmap, true_dirmap_idx, sigma=1.5)
        enh_loss = (0.5 * self.mse_criterion(pred_orig, true_orig) + 0.5 * self.bce_criterion(pred_bin, true_bin))
        total_loss = ori_loss + enh_loss
        return {"ori_loss": ori_loss, "enh_loss": enh_loss, "total_loss": total_loss}
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None: # ⚠️ Note que agora não retorna mais a loss
        # Acessa os otimizadores
        opt_dirmap, opt_enh = self.optimizers()

        # Calcula as losses
        losses = self.model_step(batch)
        ori_loss = losses["ori_loss"]
        enh_loss = losses["enh_loss"]
        total_loss = losses["total_loss"]

        # Fase 1: Aquecimento (otimiza apenas dirmap)
        if self.current_epoch < self.hparams.warmup_epochs_dirmap:
            # Zera os gradientes do otimizador específico
            opt_dirmap.zero_grad()
            # Faz o backpropagation (o Lightning cuida de escalar a loss para precisão mista, etc.)
            self.manual_backward(ori_loss)
            # Atualiza os pesos
            opt_dirmap.step()

            # Loga a métrica
            self.train_ori_loss(ori_loss)
            self.log("train/ori_loss", self.train_ori_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Fase 2: Treinamento conjunto (otimiza ambos)
        else:
            # Zera os gradientes de AMBOS os otimizadores
            opt_dirmap.zero_grad()
            opt_enh.zero_grad()
            # O backpropagation na loss total calcula gradientes para toda a rede
            self.manual_backward(total_loss)
            # Atualiza os pesos de AMBAS as sub-redes
            opt_dirmap.step()
            opt_enh.step()

            # Loga as métricas
            self.train_loss(total_loss)
            self.train_ori_loss(ori_loss)
            self.train_enh_loss(enh_loss)
            self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/ori_loss", self.train_ori_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/enh_loss", self.train_enh_loss, on_step=False, on_epoch=True, sync_dist=True)
        
    def on_train_epoch_start(self):
        if self.current_epoch < self.hparams.warmup_epochs_dirmap:
            self.net.dirmap_net.requires_grad_(True); self.net.enhancer_net.requires_grad_(False)
        elif self.current_epoch == self.hparams.warmup_epochs_dirmap:
            print(f"\nÉpoca {self.current_epoch}: Fim do aquecimento. Treinando o modelo completo!\n")
            self.net.dirmap_net.requires_grad_(True); self.net.enhancer_net.requires_grad_(True)

    def on_train_epoch_end(self) -> None: pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        losses = self.model_step(batch)
        self.val_loss(losses["total_loss"]); self.val_ori_loss(losses["ori_loss"]); self.val_enh_loss(losses["enh_loss"])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ori_loss", self.val_ori_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/enh_loss", self.val_enh_loss, on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute(); self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        losses = self.model_step(batch)
        self.test_loss(losses["total_loss"])
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self) -> None: pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit": self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        opt_dirmap = self.hparams.optimizer_dirmap(params=self.net.dirmap_net.parameters())
        opt_enh = self.hparams.optimizer_enh(params=self.net.enhancer_net.parameters())
        sched_dirmap_obj = self.hparams.scheduler_dirmap(optimizer=opt_dirmap)
        sched_dirmap = { "scheduler": sched_dirmap_obj, "monitor": "val/ori_loss", "interval": "epoch", "frequency": 1, "name": "lr/dirmap" }
        sched_enh_obj = self.hparams.scheduler_enh(optimizer=opt_enh)
        sched_enh = { "scheduler": sched_enh_obj, "monitor": "val/loss", "interval": "epoch", "frequency": 1, "name": "lr/enhancer" }
        return [opt_dirmap, opt_enh], [sched_dirmap, sched_enh]

    # ... (Seu método predict_step permanece inalterado) ...
    # <-------------------------------------------------------------------------------------->
    def save_orientation_field(self, output_tensor: torch.Tensor, mask:np.ndarray, png_path: str, dir_path: str):
        output = output_tensor.cpu().squeeze(0)
        assert output.shape[0] == 90, "Output must have 90 channels"
        max_indices = torch.argmax(output, dim=0).numpy()
        angles = max_indices * 2
        background = -np.ones_like(angles)
        if mask is not None: angles = np.where(mask == 0, background, angles)
        H, W = angles.shape
        img_array = (angles.astype(np.float32) * (255.0 / 178.0)).astype(np.uint8)
        Image.fromarray(img_array, mode="L").save(png_path)
        with open(dir_path, "w") as f:
            f.write(f"{W} {H}\n")
            for y in range(H): f.write(" ".join(str(angles[y, x]) for x in range(W)) + "\n")

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        data, names = batch[0], batch[1]
        output_paths = { 
            "gabor": os.path.join(self.output_path, "gabor"), 
            "bin": os.path.join(self.output_path, "bin"), 
            "enh": os.path.join(self.output_path, "enh"), 
            "dirmap": os.path.join(self.output_path, "dirmap"), 
            "dirmap_png": os.path.join(self.output_path, "dirmap_png"), 
            "mask": os.path.join(self.output_path, "mask") 
        }
        for path in output_paths.values(): os.makedirs(path, exist_ok=True)
        if not self.use_patches:
            dirmap_pred, latent_enh = self.forward(data)
        else: # Lógica de patches aqui...
            pass
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
            gabor.save(output_paths['gabor'] + '/' + name + '.png')

            bin = Image.fromarray(bin)
            bin.save(output_paths['bin'] + '/' + name + '.png')

            orig = Image.fromarray(orig)
            orig.save(output_paths['enh'] + '/' + name + '.png')

            dirmap   = dirmap_pred[i, :, :, :]
            dirmap   = torch.nn.functional.sigmoid(dirmap)

            # mask    = seg_pred[i, 0, :, :]
            # mask = torch.nn.functional.sigmoid(mask)
            # mask = torch.round(mask)
            # mask = mask.cpu().numpy()

            self.save_orientation_field(dirmap, None, f"{output_paths['dirmap_png']}/{name}.png", f"{output_paths['dirmap']}/{name}.dir")

    # <-------------------------------------------------------------------------------------->