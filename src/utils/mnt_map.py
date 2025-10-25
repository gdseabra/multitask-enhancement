import os
import torch
import torchvision.io as io
import torch.nn.functional as F
from tqdm import tqdm

def load_binary_image(path, device):
    img = io.read_image(path)[0]
    img = (img > 0).float().to(device)
    return img

def save_heatmap(img, path):
    img = (img / img.max()).clamp(0, 1) * 255
    io.write_png(img.byte().unsqueeze(0).cpu(), path)

def get_crossing_number_map(skel):
    device = skel.device
    skel = F.pad(skel, (1, 1, 1, 1), mode='constant', value=0)

    neighbors = [
        (-1,  0), (-1,  1), ( 0,  1), ( 1,  1),
        ( 1,  0), ( 1, -1), ( 0, -1), (-1, -1)
    ]
    neighbors_tensors = [torch.roll(skel, shifts=(dy, dx), dims=(0, 1)) for dy, dx in neighbors]

    cn = torch.zeros_like(skel)
    for i in range(8):
        u = neighbors_tensors[i]
        v = neighbors_tensors[(i + 1) % 8]
        cn += (u != v).float()

    cn = cn / 2
    cn = cn[1:-1, 1:-1]
    mask = (skel[1:-1, 1:-1] == 1) & ((cn == 1) | (cn >= 3))
    return mask.nonzero(as_tuple=False)  # [N, 2] tensor with (y, x) of valid points

def mask_black_border(mask, radius=8):
    black = (mask == 0).float().unsqueeze(0).unsqueeze(0)
    y, x = torch.meshgrid(torch.arange(-radius, radius+1), torch.arange(-radius, radius+1), indexing='ij')
    kernel = ((x**2 + y**2) <= radius**2).float().to(mask.device).unsqueeze(0).unsqueeze(0)
    dilated = F.conv2d(black, kernel, padding=radius)
    return (dilated.squeeze() == 0).float()

def mask_image_border(shape, border):
    H, W = shape
    mask = torch.zeros((H, W), dtype=torch.bool, device='cuda')
    mask[border:H-border, border:W-border] = 1
    return mask.float()

def generate_gaussian_kernel(size, sigma, device):
    ax = torch.arange(size, device=device) - size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.max()

def add_gaussians_to_heatmap(points, shape, kernel, heatmap):
    k = kernel.shape[0] // 2
    H, W = shape
    for y, x in points:
        y, x = int(y.item()), int(x.item())
        y0, y1 = max(0, y - k), min(H, y + k + 1)
        x0, x1 = max(0, x - k), min(W, x + k + 1)

        ky0, ky1 = k - (y - y0), k + (y1 - y)
        kx0, kx1 = k - (x - x0), k + (x1 - x)

        heatmap[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1]
    return heatmap

def process_directory(input_dir, mask_dir, output_dir, sigma=2.0, kernel_size=15, border=8, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)
    filenames = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]

    kernel = generate_gaussian_kernel(kernel_size, sigma, device)

    for filename in tqdm(filenames, desc="Generating Gaussian heatmaps"):
        skel_path = os.path.join(input_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        out_path = os.path.join(output_dir, filename)

        skel = load_binary_image(skel_path, device)
        mask = load_binary_image(mask_path, device)

        points = get_crossing_number_map(skel)

        valid_mask_area = mask_black_border(mask, radius=8)
        valid_border_area = mask_image_border(skel.shape, border=border)
        valid_area = (valid_mask_area * valid_border_area).bool()

        # Filter points not in valid area
        points = torch.stack([points[:, 0], points[:, 1]], dim=1)
        keep = valid_area[points[:, 0], points[:, 1]]
        points = points[keep]

        heatmap = torch.zeros_like(skel)
        heatmap = add_gaussians_to_heatmap(points, skel.shape, kernel, heatmap)

        save_heatmap(heatmap, out_path)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Gaussian heatmaps from crossing numbers.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with input skeletons (.png)')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory with masks (.png)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save heatmaps')
    parser.add_argument('--sigma', type=float, default=8.0, help='Standard deviation of Gaussian')
    parser.add_argument('--kernel_size', type=int, default=35, help='Size of Gaussian kernel')
    parser.add_argument('--border', type=int, default=8, help='Pixels from border to exclude')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')

    args = parser.parse_args()
    process_directory(
        input_dir=args.input_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        sigma=args.sigma,
        kernel_size=args.kernel_size,
        border=args.border,
        device=args.device
    )
