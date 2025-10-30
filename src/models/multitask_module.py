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

        soft_target = F.interpolate(target_one_hot, scale_factor=8, mode="bilinear", align_corners=False)

        # Use log_sigmoid for numerical stability
        # log(p) = log(sigmoid(logits)) = log_sigmoid(logits)
        log_p = F.logsigmoid(logits)
        
        # log(1-p) = log(1 - sigmoid(logits)) = log_sigmoid(-logits)
        log_one_minus_p = F.logsigmoid(-logits)

        # Calculate weighted loss terms
        loss_pos = self.lambda_pos * soft_target * log_p
        loss_neg = self.lambda_neg * (1 - soft_target) * log_one_minus_p
        
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

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, true_dirmap_idx, y_orig, y_bin, mask = batch
        
        # --- Forward ---
        # pred_dirmap são os logits (B, 90, H, W)
        pred_dirmap, pred_enh = self.forward(x)
        
        # --- Preparar Predições ---
        pred_orig, pred_bin = pred_enh[:,0,:,:], pred_enh[:,1,:,:]
        # Converte logits do dirmap para probabilidades (necessário para as novas losses)
        # pred_dirmap_probs = torch.sigmoid(pred_dirmap)

        # --- Preparar Targets ---
        true_orig, true_bin = y_orig[:, 0, :, :], y_bin[:, 0, :, :]

        
        # --- Preparar Inputs para Loss de Orientação ---
        # Labels precisam ter dimensão de canal: (B, H, W) -> (B, 1, H, W)
        true_dirmap_labels = true_dirmap_idx
        # Usar 'true_dirmap_labels==90' como a máscara ROI: (B, H, W) -> (B, 1, H, W)
        roi_mask = (mask > 0).float()

        roi_mask_small = F.interpolate(roi_mask, scale_factor=1/8, mode="nearest").long()

        # remove values de máscara
        true_dirmap_labels[roi_mask_small == 0] = 0

        # --- Calcular Losses ---
        
        # 1. Loss de Orientação (NOVA LÓGICA)
        loss_ori_weighted = self.orientation_loss_fn(pred_dirmap, true_dirmap_labels, roi_mask)
        loss_ori_coherence = self.coherence_loss_fn(pred_dirmap, roi_mask)
        
        # Soma ponderada dos componentes da loss de orientação
        ori_loss = (self.hparams.w_ori * loss_ori_weighted) + \
                   (self.hparams.w_coh * loss_ori_coherence)
        
        # 2. Loss de Enhancement (Inalterada)
        enh_loss = (0.5 * self.mse_criterion(pred_orig, true_orig) + \
                    0.5 * self.bce_criterion(pred_bin, true_bin))
        
        # 3. Loss Total (Inalterada)
        total_loss = ori_loss + enh_loss
        
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