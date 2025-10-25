import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append('..')
from model.my_network import FingerGAN
from util.my_data_loader import LatentDataset
from loss.loss import adversarial_loss, reconstruction_loss  # Assume these are defined

class TrainNetwork:
    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.IN_CHANNELS = 1
        self.INPUT_SIZE = (192, 192)
        self.STEP = 10
        self.EPOCHS = 50
        self.BATCH_SIZE = 8
        self.LEARNING_RATE = 1e-4

        # Dummy Data (replace with actual tensors)
        dummy_latent = torch.rand(384, 384)  # Replace with actual latent tensor
        dummy_reference = torch.rand(384, 384)  # Replace with ground truth reference tensor

        latent_dataset = LatentDataset(dummy_latent, self.INPUT_SIZE[0], self.INPUT_SIZE[1], self.STEP)
        reference_dataset = ReferenceDataset(dummy_reference, self.INPUT_SIZE[0], self.INPUT_SIZE[1], self.STEP)

        assert len(latent_dataset) == len(reference_dataset), "Latent and reference datasets must have the same length"

        self.train_loader = DataLoader(list(zip(latent_dataset, reference_dataset)), batch_size=self.BATCH_SIZE, shuffle=True)

        self.G = FingerGAN(in_channels=self.IN_CHANNELS).to(self.DEVICE)
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=self.LEARNING_RATE)
        self.criterion = MyWeightedL1Loss()

    def train(self):
        for epoch in range(self.EPOCHS):
            self.G.train()
            for i, batch in enumerate(self.train_loader):
                (row_col_latent, latent), (row_col_ref, reference) = zip(*batch)
                latent = torch.stack(latent).to(self.DEVICE)
                reference = torch.stack(reference).to(self.DEVICE)

                pixel_weight = torch.ones_like(reference).to(self.DEVICE)  # Use actual weights if available

                self.optimizer_G.zero_grad()
                output = self.G(latent)
                loss_G = self.criterion(output, reference, pixel_weight)
                loss_G.backward()
                self.optimizer_G.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{self.EPOCHS}] [Batch {i}/{len(self.train_loader)}] "
                          f"[Reconstruction loss: {loss_G.item():.4f}]")

            torch.save(self.G.state_dict(), f"checkpoint_G_epoch{epoch}.pt")


if __name__ == '__main__':
    train_network = TrainNetwork()
    train_network.train()

