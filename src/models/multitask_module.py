import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
from PIL import Image
from typing import Callable, Tuple, Dict, Any
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.aggregation import MinMetric

# ----------------------------------------------------------------------------
# --- Definições das Novas Classes de Loss ---
# ----------------------------------------------------------------------------

class WeightedOrientationLoss(nn.Module):
    def __init__(self, lambda_pos: float = 1.0, lambda_neg: float = 0.25):
        super().__init__()
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw output from the model (B, N, H, W).
                                   DO NOT apply sigmoid beforehand.
            target (torch.Tensor): Ground truth labels (B, 1, H, W).
            mask (torch.Tensor): ROI mask (B, 1, H, W).
        """
        num_classes = logits.shape[1]
        
        # Create one-hot target
        target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Use log_sigmoid for numerical stability
        # log(p) = log(sigmoid(logits)) = log_sigmoid(logits)
        log_p = F.logsigmoid(logits)
        
        # log(1-p) = log(1 - sigmoid(logits)) = log_sigmoid(-logits)
        log_one_minus_p = F.logsigmoid(-logits)

        # Calculate weighted loss terms
        loss_pos = self.lambda_pos * target_one_hot * log_p
        loss_neg = self.lambda_neg * (1 - target_one_hot) * log_one_minus_p
        
        loss = -(loss_pos + loss_neg)

        # Apply mask and normalize by the number of pixels in the ROI
        mask = mask.float()
        masked_loss = loss * mask
        
        num_roi_pixels = mask.sum()
        if num_roi_pixels > 0:
            total_loss = masked_loss.sum() / num_roi_pixels
        else:
            total_loss = torch.tensor(0.0, device=logits.device)

        return total_loss

# ----------------------------------------------------------------------------

class OrientationCoherenceLoss(nn.Module):
    """
    Implementa a loss de coerência de orientação (L_odpi) do artigo.
    
    L_odpi = |ROI| / (sum_{ROI} Coh) - 1
    
    Onde Coh é o mapa de coerência calculado dos vetores de orientação preditos.
    """
    def __init__(self, N: int = 90, epsilon: float = 1e-8):
        """
        Args:
            N (int): O número de ângulos de orientação discretos.
            epsilon (float): Valor pequeno para estabilidade numérica.
        """
        super().__init__()
        self.N = N
        self.epsilon = epsilon

        # Pré-computa ângulos e o kernel 3x3 (J_3)
        angle_step_deg = 180.0 / N
        angles_deg = torch.arange(N, dtype=torch.float32) * angle_step_deg
        
        # A fórmula usa cos(2 * angulo) e sin(2 * angulo)
        angles_rad_doubled = torch.deg2rad(2 * angles_deg)

        # Registra como 'parameter' (buffers não-treináveis)
        self.cos_terms = nn.Parameter(torch.cos(angles_rad_doubled).view(1, N, 1, 1), requires_grad=False)
        self.sin_terms = nn.Parameter(torch.sin(angles_rad_doubled).view(1, N, 1, 1), requires_grad=False)
        self.j3_kernel = nn.Parameter(torch.ones((1, 1, 3, 3), dtype=torch.float32), requires_grad=False)

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction (torch.Tensor): Mapa de probabilidade do modelo. (B, N, H, W).
            mask (torch.Tensor): Máscara ROI. (B, 1, H, W).
                                 
        Returns:
            torch.Tensor: O valor escalar da loss.
        """

        prediction = torch.sigmoid(logits)
        # --- 1. Computa o vetor de orientação médio d_bar ---
        d_bar_cos = torch.sum(prediction * self.cos_terms, dim=1, keepdim=True) / self.N
        d_bar_sin = torch.sum(prediction * self.sin_terms, dim=1, keepdim=True) / self.N
        
        # --- 2. Calcula Coh = (d_bar * J_3) / (|d_bar| * J_3) ---
        
        # Numerador: Convolui componentes do vetor e calcula magnitude do resultado
        summed_d_cos = F.conv2d(d_bar_cos, self.j3_kernel, padding='same')
        summed_d_sin = F.conv2d(d_bar_sin, self.j3_kernel, padding='same')
        numerator = torch.sqrt(summed_d_cos**2 + summed_d_sin**2 + self.epsilon)
        
        # Denominador: Calcula magnitude dos vetores, depois convolui (média)
        d_bar_mag = torch.sqrt(d_bar_cos**2 + d_bar_sin**2 + self.epsilon)
        denominator = F.conv2d(d_bar_mag, self.j3_kernel, padding='same')
        
        coh = numerator / (denominator + self.epsilon)
        
        # --- 3. Computa a loss final L_odpi ---
        mask = mask.float()
        roi_size = torch.sum(mask)
        
        if roi_size > 0:
            sum_coh_roi = torch.sum(coh * mask)
            loss = roi_size / (sum_coh_roi + self.epsilon) - 1.0
        else:
            loss = torch.tensor(0.0, device=prediction.device, requires_grad=True)
            
        return loss

# ----------------------------------------------------------------------------
# --- Módulo Lightning Atualizado ---
# ----------------------------------------------------------------------------

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
        warmup_epochs_dirmap: int = 2,
        # --- NOVOS Hiperparâmetros para a Loss ---
        w_ori: float = 1.0,         # Peso para a loss de orientação ponderada
        w_coh: float = 0.5,         # Peso para a loss de coerência
        lambda_pos: float = 1.0,    # Peso positivo para WeightedOrientationLoss
        lambda_neg: float = 0.25,   # Peso negativo para WeightedOrientationLoss
        N_ori: int = 90             # Número de classes de orientação
    ) -> None:
        super().__init__()
        # Salva todos os HPs, incluindo os novos
        self.save_hyperparameters(logger=False, ignore=["net"]) 
        self.net = net

        self.automatic_optimization = False

        self.mse_criterion = torch.nn.functional.mse_loss
        self.bce_criterion = torch.nn.functional.binary_cross_entropy_with_logits
        
        # --- NOVAS Funções de Loss ---
        self.orientation_loss_fn = WeightedOrientationLoss(
            lambda_pos=self.hparams.lambda_pos, 
            lambda_neg=self.hparams.lambda_neg
        )
        self.coherence_loss_fn = OrientationCoherenceLoss(N=self.hparams.N_ori)
        
        # --- Métricas ---
        self.train_loss = MeanMetric()
        self.train_ori_loss = MeanMetric()
        self.train_enh_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_ori_loss = MeanMetric()
        self.val_enh_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        
        # --- Outros Atributos ---
        self.patch_size = patch_size
        self.use_patches = use_patches
        self.stride = stride
        self.output_path = output_path


    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)
    
    def on_train_start(self) -> None:
        self.val_loss.reset(); self.val_ori_loss.reset(); self.val_enh_loss.reset(); self.val_loss_best.reset()

    # Defines a single model step (e.g., for one training or validation batch)
    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # Unpacks the batch tuple into its components.
        # x: Input tensor (e.g., images), shape (B, C_in, H_small, W_small)
        # true_dirmap_labels: Ground truth orientation labels at a small resolution
        #                     Shape: (B, 1, H/8, W/8)
        # y_orig, y_bin: Ground truth for enhancement (original and binary)
        # roi_mask: Region of Interest mask, full resolution, Shape: (B, 1, H, W)
        x, true_dirmap_labels, y_orig, y_bin, roi_mask = batch
        
        # --- Forward ---
        # Runs the model's forward pass on the input 'x'
        # pred_dirmap: Predicted orientation logits. (B, 90, H, W)
        # pred_enh: Predicted enhancement, likely (B, 2, H, W)
        # inter_gabor: Intermediate Gabor features (B, 1, H, W)
        pred_dirmap, pred_enh, inter_gabor = self.forward(x)
        
        # --- Preparar Predições ---
        # Prepares (unpacks) the predictions for loss calculation
        
        # Extracts the 'original' (grayscale) prediction from channel 0
        # Extracts the 'binary' (mask) prediction from channel 1
        pred_orig, pred_bin = pred_enh[:,0,:,:], pred_enh[:,1,:,:]

        # --- Prepare Targets ---
        # Prepares (unpacks) the ground truth targets for loss calculation
        
        # Extracts 'original' and 'binary' ground truth, removing channel dim
        true_orig, true_bin = y_orig[:, 0, :, :], y_bin[:, 0, :, :]
        
        # Downsamples the full-resolution ROI mask to match the initial label map size (1/8th scale)
        # Output Shape: (B, 1, H/8, W/8)
        small_dirmap_mask = F.interpolate(roi_mask, scale_factor=1/8, mode="bilinear") 

        # --- Start of Complex Target Preparation ---
        # The goal is to convert low-res angle labels (0-179) into full-res
        # class indices (0-89) and a background mask.
        
        # 'indices' holds the low-res angle labels. Shape: (B, 1, H/8, W/8)
        indices = true_dirmap_labels 
        
        # Finds all pixels with odd-numbered angles
        is_odd = indices % 2 != 0
        # Rounds odd angles down (e.g., 179 -> 178)
        indices[is_odd] -= 1
        # Maps angle ranges [0,1], [2,3]...[178,179] to classes 0, 1... 89
        indices = indices // 2
        # 'indices' now holds classes 0-89. Shape: (B, 1, H/8, W/8)
        
        # Sets all pixels *outside* the downsampled ROI to class 90
        # Class 90 will serve as the "ignore" or "background" class
        indices[small_dirmap_mask == 0] = 90
        
        # Initializes a zero tensor for one-hot encoding
        # Shape: (B, 91, H/8, W/8) (90 classes + 1 'ignore' class)
        one_hot = torch.zeros(indices.shape[0], 91, indices.shape[-2], indices.shape[-1], dtype=torch.float32, device=indices.device)

        # Populates the 'one_hot' tensor. 
        # For each pixel, it places a 1.0 at the channel index specified by 'indices'.
        # 'indices' shape (B, 1, H/8, W/8) is broadcast-compatible for scatter.
        one_hot.scatter_(dim=1, index=indices, value=1.0)
        
        # 'orientation_one_hot' is the low-res one-hot map. Shape: (B, 91, H/8, W/8)
        orientation_one_hot = one_hot

        # Upsamples the one-hot target map by 8x (from H/8, W/8 to H, W)
        # This matches the full resolution of the prediction 'pred_dirmap'
        # Bilinear interpolation creates "soft" boundaries between classes.
        # Output Shape: (B, 91, H, W)
        true_dirmap = F.interpolate(orientation_one_hot, scale_factor=8, mode="bilinear", align_corners=False)
        
        # Converts the "soft" upsampled map back to "hard" class indices
        # by taking the argmax along the channel (class) dimension.
        # Output Shape: (B, H, W)
        true_dirmap_idx = true_dirmap.argmax(dim=1)

        # Re-assigns 'true_dirmap_labels' to this new full-resolution index map
        # Adds a channel dimension back.
        # Output Shape: (B, 1, H, W). 
        true_dirmap_labels = true_dirmap_idx.unsqueeze(1)
        
        # Creates a binary mask. Pixels are 1 (keep) if they are NOT the
        # 'ignore' class (90), and 0 (ignore) if they are.
        # Output Shape: (B, 1, H, W). 
        dirmap_mask = (true_dirmap_labels != 90).long()
        
        # Sets the 'ignore' class pixels (90) to 0.
        # This is safe because 'dirmap_mask' already knows to ignore them.
        # This step makes the labels (0-89) compatible with the
        # 90-class prediction 'pred_dirmap'.
        true_dirmap_labels[true_dirmap_labels == 90] = 0
        # --- End of Target Preparation ---

        
        # Calculates the main orientation loss (e.g., CrossEntropy)
        # The loss function likely uses 'dirmap_mask' to ignore background pixels.
        loss_ori_weighted = self.orientation_loss_fn(pred_dirmap, true_dirmap_labels, dirmap_mask)
        
        # Calculates a coherence/regularization loss for orientation
        loss_ori_coherence = self.coherence_loss_fn(pred_dirmap, dirmap_mask)
        
        # Combines the orientation losses, weighted by hyperparameters
        ori_loss = (self.hparams.w_ori * loss_ori_weighted) + \
                   (self.hparams.w_coh * loss_ori_coherence)
        
        # Calculates the enhancement loss
        # 50% MSE loss for the 'original' (grayscale) regression task
        # 50% BCE loss for the 'binary' (mask) segmentation task
        enh_loss = (0.5 * self.mse_criterion(pred_orig, true_orig) + \
                    0.5 * self.bce_criterion(pred_bin, true_bin))
        
        # Sums the two main loss components to get the final loss
        total_loss = ori_loss + enh_loss
        
        # Returns a dictionary of losses for logging
        return {"ori_loss": ori_loss, "enh_loss": enh_loss, "total_loss": total_loss}
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None: 
        opt_dirmap, opt_enh = self.optimizers()

        losses = self.model_step(batch)
        ori_loss = losses["ori_loss"]
        enh_loss = losses["enh_loss"]
        total_loss = losses["total_loss"]

        # Fase 1: Aquecimento (otimiza apenas dirmap)
        if self.current_epoch < self.hparams.warmup_epochs_dirmap:
            opt_dirmap.zero_grad()
            self.manual_backward(ori_loss)
            opt_dirmap.step()
            self.train_ori_loss(ori_loss)
            self.log("train/ori_loss", self.train_ori_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Fase 2: Treinamento conjunto (otimiza ambos)
        else:
            opt_dirmap.zero_grad()
            opt_enh.zero_grad()
            self.manual_backward(total_loss)
            opt_dirmap.step()
            opt_enh.step()

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
            "mask": os.path.join(self.output_path, "mask"),
            "gabor_dirmap": os.path.join(self.output_path, "inter_gabor")
        }
        for path in output_paths.values(): os.makedirs(path, exist_ok=True)
        if not self.use_patches:
            dirmap_pred, latent_enh, inter_gabor = self.forward(data)
        else: # Lógica de patches aqui...
            pass
        for i, name in enumerate(names):
            name = name.split('/')[-1].split('.')[0]
            gabor   = latent_enh[i, 1, :, :]
            orig    = latent_enh[i, 0, :, :]
            gabor_dirmap = inter_gabor[i, 0, :, :]

            gabor   = torch.nn.functional.sigmoid(gabor)
            bin   = torch.round(gabor)

            gabor = gabor.cpu().numpy()
            bin   = bin.cpu().numpy()
            orig  = orig.cpu().numpy()

            gabor_dirmap = gabor_dirmap.cpu().numpy()

            gabor = (255 * (gabor - np.min(gabor))/(np.max(gabor) - np.min(gabor))).astype('uint8')
            bin   = (255 * (bin - np.min(bin))/(np.max(bin) - np.min(bin))).astype('uint8')
            orig   = (255 * (orig - np.min(orig))/(np.max(orig) - np.min(orig))).astype('uint8')
            gabor_dirmap = (255 * (gabor_dirmap - np.min(gabor_dirmap))/(np.max(gabor_dirmap) - np.min(gabor_dirmap))).astype('uint8')

            gabor = Image.fromarray(gabor)
            gabor.save(output_paths['gabor'] + '/' + name + '.png')

            gabor_dirmap = Image.fromarray(gabor_dirmap)
            gabor_dirmap.save(output_paths['gabor_dirmap'] + '/' + name + '.png')

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