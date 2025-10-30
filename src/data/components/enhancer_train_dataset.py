import os

import torch
import wsq
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import numpy as np

import os
import torch
import wsq
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np

class EnhancerTrainDataset(Dataset):
    def __init__(self, 
        data_dir: str = "data/", 
        data_list: str = None, 
        transform=None, 
        skel_transform=None, 
        patch_size=None, 
        lat_subdir = '/latents/', 
        ref_subdir = '/references/', 
        skel_subdir = '/skel/', 
        bin_subdir = '/bin/', 
        mask_subdir = '/masks/', 
        ref_mask_subdir = '/ref_masks/',
        mnt_subdir = '/mnt/', 
        orient_subdir = '/orient/',   # NEW: directory with .dir files
        apply_mask = True,
        use_ref_mask = False
    ):
        self.data_dir        = data_dir
        self.transform       = transform
        self.skel_transform  = skel_transform
        self.data_list       = data_dir + data_list if data_list is not None else data_dir + "/data_list.txt"

        with open(self.data_list) as fp:
            lines = fp.readlines()

        self.data = [line.strip() for line in lines]

        self.lat_suffix   = "." + os.listdir(data_dir + lat_subdir)[0].split(".")[-1]
        self.ref_suffix   = "." + os.listdir(data_dir + ref_subdir)[0].split(".")[-1]
        self.skel_suffix  = "." + os.listdir(data_dir + skel_subdir)[0].split(".")[-1]
        self.bin_suffix   = "." + os.listdir(data_dir + bin_subdir)[0].split(".")[-1]
        self.mnt_suffix   = "." + os.listdir(data_dir + mnt_subdir)[0].split(".")[-1]
        self.mask_suffix  = "." + os.listdir(data_dir + mask_subdir)[0].split(".")[-1]
        self.orient_suffix = "." + os.listdir(data_dir + orient_subdir)[0].split(".")[-1]

        self.lat_subdir   = lat_subdir
        self.ref_subdir   = ref_subdir
        self.skel_subdir  = skel_subdir
        self.bin_subdir   = bin_subdir
        self.mask_subdir  = mask_subdir
        self.ref_mask_subdir  = ref_mask_subdir
        self.mnt_subdir   = mnt_subdir
        self.orient_subdir = orient_subdir

        self.apply_mask   = apply_mask
        self.use_ref_mask = use_ref_mask
        self.patch_size   = patch_size


    def load_orientation_field(self, filepath: str, mask: Image) -> torch.Tensor:
        """Reads .dir file and returns a one-hot tensor of size (90, H/8, W/8)."""
        with open(filepath, "r") as f:
            lines = f.readlines()


        # First line: width height
        width, height = map(int, lines[0].split())
        n_blocks_w, n_blocks_h = width // 8, height // 8

        # Read the block orientations
        raw_vals = []
        for line in lines[1:]:
            raw_vals.extend(map(int, line.split()))

        raw_vals = np.array(raw_vals).reshape(n_blocks_h, n_blocks_w)

        mask = mask.resize((mask.width // 8, mask.height // 8), Image.Resampling.BILINEAR)
        mask = np.array(mask, dtype=np.uint8)

        # One-hot encoding with 90 channels
        one_hot = np.zeros((91, n_blocks_h, n_blocks_w), dtype=np.float32)

        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                angle = raw_vals[i, j]
                if angle < 0 or mask[i,j] == 0.0:
                    one_hot[90, i, j] = 1.0
                    continue  # leave as all zeros
                if angle % 2 != 0:
                    angle -= 1  # floor to nearest even
                idx = angle // 2
                one_hot[idx, i, j] = 1.0


        return torch.tensor(one_hot)


    def __getitem__(self, ix):
        lat   = Image.open(self.data_dir+self.lat_subdir+self.data[ix]+self.lat_suffix)

        ref_mask  = Image.open(self.data_dir + self.ref_mask_subdir + self.data[ix].split('_')[0] + self.mask_suffix)

        if not self.use_ref_mask:
            mask  = Image.open(self.data_dir + self.mask_subdir  + self.data[ix] + self.mask_suffix)

        try:
            ref   = Image.open(self.data_dir + self.ref_subdir   + self.data[ix] + self.ref_suffix)
            bin   = Image.open(self.data_dir + self.bin_subdir   + self.data[ix] + self.bin_suffix)
            skel  = Image.open(self.data_dir + self.skel_subdir  + self.data[ix].split('_')[0] + self.skel_suffix)
            if self.use_ref_mask:
                mask = Image.open(self.data_dir + self.mask_subdir + self.data[ix].split('_')[0] + self.mask_suffix)

        except FileNotFoundError:
            ref   = Image.open(self.data_dir + self.ref_subdir   + self.data[ix].split('_')[0] + self.ref_suffix)
            bin   = Image.open(self.data_dir + self.bin_subdir   + self.data[ix].split('_')[0] + self.bin_suffix)
            skel  = Image.open(self.data_dir + self.skel_subdir  + self.data[ix].split('_')[0] + self.skel_suffix)
            if self.use_ref_mask:
                mask = Image.open(self.data_dir + self.mask_subdir + self.data[ix].split('_')[0] + self.mask_suffix)


        # --- Normalize lat & ref ---
        lat = transforms.ToTensor()(lat)
        ref = transforms.ToTensor()(ref)

        lat_mean, lat_std = torch.mean(lat), torch.std(lat)
        eps = 1e-6
        lat = transforms.Normalize(mean=[lat_mean], std=[2 * lat_std + eps])(lat)

        ref_mean, ref_std = torch.mean(ref), torch.std(ref)
        ref = transforms.Normalize(mean=[ref_mean], std=[2 * ref_std])(ref)

        # --- Orientation field (label) ---
        orient_file = self.data_dir + self.orient_subdir + self.data[ix].split('_')[0] + self.orient_suffix
        # orientation_one_hot = self.load_orientation_field(orient_file, mask)
        orient_img = Image.open(orient_file).convert("L")
        to_tensor = transforms.ToTensor()
        dirmap_target = to_tensor(orient_img)

        # 2. Rescale tensor values back to the original [0, 180] range
        # We multiply by 255 because ToTensor() divides by 255.
        dirmap_target *= 255.0

        # 3. Round values to the nearest even number
        # (e.g., 25.1 -> 26, 24.9 -> 24)
        dirmap_target = torch.round(dirmap_target)

        # 4. Map the value 180 to 0 using the modulo operator
        # This leaves all other values unchanged (e.g., 178 % 180 = 178)
        dirmap_target_idx = (dirmap_target % 180)//2  # Now in [0,89]

        # 5. Convert to an integer tensor for clean, discrete values
        dirmap_target_idx = dirmap_target_idx.long()

        # Apply skeleton transform
        bin  = self.skel_transform(bin)
        mask = self.skel_transform(mask)
        ref_mask = self.skel_transform(ref_mask)
        skel = self.skel_transform(skel)

        ref_white, bin_white, lat_white = ref.max(), bin.max(), lat.max()

        if self.apply_mask:
            ref  = torch.where(mask == 0, ref_white, ref)
            bin  = torch.where(mask == 0, bin_white, bin)

        mask = mask*ref_mask
        return lat, dirmap_target_idx, ref, bin, mask

    def __len__(self):
        return len(self.data)

