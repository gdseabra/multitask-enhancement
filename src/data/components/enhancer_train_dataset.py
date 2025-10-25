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
                    angle -= 1  # round down to nearest even
                idx = angle // 2
                one_hot[idx, i, j] = 1.0


        return torch.tensor(one_hot)


    def __getitem__(self, ix):
        lat   = Image.open(self.data_dir+self.lat_subdir+self.data[ix]+self.lat_suffix)

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
        orientation_one_hot = self.load_orientation_field(orient_file, mask)

        # Apply skeleton transform
        bin  = self.skel_transform(bin)
        mask = self.skel_transform(mask)
        skel = self.skel_transform(skel)

        ref_white, bin_white, lat_white = ref.max(), bin.max(), lat.max()

        if self.apply_mask:
            ref  = torch.where(mask == 0, ref_white, ref)
            bin  = torch.where(mask == 0, bin_white, bin)

        return lat, orientation_one_hot, ref, bin

    def __len__(self):
        return len(self.data)



# class PatchEnhancerTrainDataset(Dataset):
#     def __init__(self, 
#         data_dir: str = "data/", 
#         data_list: str = None, 
#         transform=None, 
#         skel_transform=None, 
#         patch_size=None, 
#         lat_subdir = '/latents/', 
#         ref_subdir = '/references/', 
#         skel_subdir = '/skel/', 
#         bin_subdir = '/bin/', 
#         mask_subdir = '/masks/', 
#         mnt_subdir='/mnt/', 
#         apply_mask = 0, 
#         patch_shape = (128,128), 
#         stride = 128
#     ):
#         self.input_row = patch_shape[0]
#         self.input_col = patch_shape[1]


#         self.data_dir        = data_dir
#         self.transform       = transform
#         self.skel_transform = skel_transform
#         self.data_list       = data_dir + data_list if data_list is not None else data_dir + "/data_list.txt"

#         with open(self.data_list) as fp:
#             lines = fp.readlines()

#         self.data = [line.strip() for line in lines]

#         self.lat_suffix   = "." + os.listdir(data_dir + lat_subdir)[0].split(".")[-1]
#         self.ref_suffix   = "." + os.listdir(data_dir + ref_subdir)[0].split(".")[-1]
#         self.skel_suffix = "." + os.listdir(data_dir + skel_subdir)[0].split(".")[-1]
#         self.bin_suffix = "." + os.listdir(data_dir + bin_subdir)[0].split(".")[-1]
#         self.mnt_suffix = "." + os.listdir(data_dir + mnt_subdir)[0].split(".")[-1]
#         self.mask_suffix  = "." + os.listdir(data_dir + mask_subdir)[0].split(".")[-1]
        
#         self.lat_subdir   = lat_subdir
#         self.ref_subdir   = ref_subdir
#         self.skel_subdir = skel_subdir
#         self.bin_subdir = bin_subdir
#         self.mask_subdir  = mask_subdir
#         self.mnt_subdir = mnt_subdir

#         self.apply_mask = apply_mask

#         self.patch_size = patch_size

#         sample_latent = Image.open(self.data_dir+self.lat_subdir+self.data[0]+self.lat_suffix)

#         shape_latent = sample_latent.size
#         ROW = shape_latent[1]
#         COL = shape_latent[0]
#         row_list_1 = range(self.input_row, ROW+1, stride)
#         row_list_2 = range(ROW, row_list_1[-1]-1,-stride)
#         row_list = [*row_list_1, *row_list_2]
        
#         col_list_1 = range(self.input_col, COL+1, stride)
#         col_list_2 = range(COL, col_list_1[-1]-1, -stride)
#         col_list = [*col_list_1,*col_list_2]
        
#         self.num_patch = len(self.data) * len(row_list)*len(col_list)

#         row_col_inds = np.zeros([self.num_patch,2]).astype(np.int32)


#         self.patches_dict = {}
#         patch_ind = 0
#         for img_i in range(len(self.data)):
#             for row_ind in row_list:
#                 for col_ind in col_list:
#                     row_col_inds[patch_ind,:] = [row_ind,col_ind]
#                     self.patches_dict[patch_ind] = (self.data[img_i], (row_ind, col_ind))
#                     patch_ind += 1


#     def __len__(self):
#         return self.num_patch

#     def read_mnt(self, file_name, threshold_score = 0.0):
#         f = open(file_name)
#         minutiae = []
#         for i, line in enumerate(f):
#             if i < 2 or len(line) == 0: continue

#             w, h, o, s, *rest = [float(x) for x in line.split()]
#             w, h = int(round(w)), int(round(h))
#             if s > threshold_score:
#                 minutiae.append([w, h, o, s])
#         f.close()
#         return np.array(minutiae)

#     def extract_mcc_cpu(self, mnts, size, K: int, sigma: float = 5.0) -> torch.Tensor:
#         radius = 4
#         img = torch.zeros(size)
#         for (x, y, _, _) in mnts:
#             x, y = int(x), int(y)
#             img[y-radius:y+radius, x-radius:x+radius] = 1
        
#         ndim = len(size)
#         img = img.long().numpy()

#         g_size = 6 * sigma + 3
#         x = np.arange(0, g_size, 1, float)
#         y = x[:, np.newaxis]
#         x0, y0 = 3 * sigma + 1, 3 * sigma + 1
#         g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (sigma**2))
#         output  = np.zeros((K, *img.shape), dtype=np.float32)

#         for x, y, o, s in mnts:
#             c = [y,x]
#             o = o % (2 * np.pi)
            

#             # center
#             ul = tuple(int(np.round(c[i] - 3 * sigma - 1)) for i in reversed(range(ndim)))
#             br = tuple(int(np.round(c[i] + 3 * sigma + 2)) for i in reversed(range(ndim)))
#             g_crop = tuple(
#                 slice(max(0, -ul[i]), min(br[i], size[i]) - ul[i])
#                 for i in reversed(range(ndim))
#             )
#             c_crop = tuple(
#                 slice(max(0, ul[i]), min(br[i], size[i])) for i in reversed(range(ndim))
#             )
#             for b in range(K):
#                 diff       = o - b * np.pi/6
#                 abs_diff   = np.abs(diff)
#                 if diff < -np.pi or diff > np.pi:
#                     d = 2 * np.pi - abs_diff
#                 else:
#                     d = abs_diff
                
#                 w = np.exp(-d / np.pi * 6)


#                 try:
#                     output[b][c_crop] += g[g_crop] * w
#                 except:
#                     pass
                

#         return torch.from_numpy(output)

#     def __getitem__(self, ix):

#         row_col = self.patches_dict[ix][1]
#         row = row_col[0]
#         col = row_col[1]

#         lat   = Image.open(self.data_dir+self.lat_subdir+self.patches_dict[ix][0]+self.lat_suffix)
#         mask  = Image.open(self.data_dir + self.mask_subdir  + self.patches_dict[ix][0] + self.mask_suffix)

#         try:
#             ref   = Image.open(self.data_dir + self.ref_subdir   + self.patches_dict[ix][0] + self.ref_suffix)
#             bin = Image.open(self.data_dir + self.bin_subdir + self.patches_dict[ix][0] + self.bin_suffix)
#             # skel = Image.open(self.data_dir + self.skel_subdir + self.patches_dict[ix][0] + self.skel_suffix)
#             mnts = self.read_mnt(self.data_dir + self.mnt_subdir + self.patches_dict[ix][0] + self.mnt_suffix)


#         except FileNotFoundError: # especial case when are dealing with an synthetic augmented dataset
#             ref   = Image.open(self.data_dir + self.ref_subdir   + self.patches_dict[ix][0].split('_')[0] + self.ref_suffix)
#             bin = Image.open(self.data_dir + self.bin_subdir + self.patches_dict[ix][0].split('_')[0] + self.bin_suffix)
#             # skel = Image.open(self.data_dir + self.skel_subdir + self.patches_dict[ix][0].split('_')[0] + self.skel_suffix)
#             mnts = self.read_mnt(self.data_dir + self.mnt_subdir + self.patches_dict[ix][0].split('_')[0] + self.mnt_suffix)

#         mcc      = self.extract_mcc_cpu(mnts, ref.size, 12, 8)


#         # normalizing lat and ref to -1, 1

#         lat = transforms.ToTensor()(lat)
#         ref = transforms.ToTensor()(ref)
#         bin = self.skel_transform(bin)
#         mask  = self.skel_transform(mask)
        


#         # crop images to patch size and coords
#         lat = lat[:,(row-self.input_row):row,(col-self.input_col):col]
#         ref = ref[:,(row-self.input_row):row,(col-self.input_col):col]
#         bin = bin[:,(row-self.input_row):row,(col-self.input_col):col]
#         mask = mask[:,(row-self.input_row):row,(col-self.input_col):col]
#         mcc = mcc[:,(row-self.input_row):row,(col-self.input_col):col]


#         lat_mean = torch.mean(lat)
#         lat_std  = torch.std(lat)

#         epsilon = 1e-6  # or 1e-8 for smaller effect
#         lat = transforms.Normalize(mean=[lat_mean], std=[2 * lat_std + epsilon])(lat)


#         ref_mean = torch.mean(ref)
#         ref_std  = torch.std(ref)


#         ref = transforms.Normalize(mean=[ref_mean], std=[2 * ref_std + epsilon])(ref)

#         # if self.skel_transform:
#             # mnt = self.skel_transform(mnt)

#         ref_white = ref.max()
#         bin_white = bin.max()
        
#         ref   = torch.where(mask == 0, ref_white, ref)
#         bin = torch.where(mask == 0, bin_white, bin)


#         return lat, torch.concat([ref, bin, mask], axis=0)
