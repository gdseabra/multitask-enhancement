# Griaule
# Author: André Igor Nóbrega da Silva
# email :  andre.igor@griaule.com
# date  : 2023-09-19
# Generates a synthetic latent fingerprint database, applying contrast adjustments, gaussian blur, gaussian noise, occlusion, and downsampling

import sys
import os
import random
from argparse import ArgumentParser
from functools import partial
import glob
import warnings
import wsq
import cv2
import numpy as np
from PIL import Image
from skimage.measure import block_reduce
from multiprocessing import Pool, cpu_count


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def gaussian_noise(img, noise_level):
    gaussian = np.random.normal(0, noise_level, img.shape)
    noisy_img = img + gaussian
    return noisy_img

def downsampling(img, block_size):
    down = block_reduce(img, block_size=(block_size, block_size), func=np.mean)
    resized = cv2.resize(down, (img.shape[1], img.shape[0]), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
    return resized 

def gamma(image=None, value=1):
    _max = image.max()
    image = (image / _max) ** value
    image = image * _max
    image = np.clip(image, a_min=0, a_max=_max)
    return image

def center_crop(img, shape):
    width, height = img.shape
    center_x = width // 2
    center_y = height // 2

    start_x = center_x - shape[0] // 2
    end_x = center_x + shape[0] // 2
    start_y = center_y - shape[1] // 2
    end_y = center_y + shape[1] // 2

    return img[start_x:end_x, start_y:end_y]


def resize_or_pad_background(background, image):
    bg_h, bg_w = background.shape[:2]
    img_h, img_w = image.shape[:2]

    background = cv2.resize(background, (bg_w*4, bg_h*4), interpolation=cv2.INTER_CUBIC)

    if (bg_h, bg_w) == (img_h, img_w):
        return background  # Already correct size

    if bg_h > img_h or bg_w > img_w:
        # Crop the background if it's larger
        start_h = max(0, (bg_h - img_h) // 2)
        start_w = max(0, (bg_w - img_w) // 2)
        background = background[start_h:start_h + img_h, start_w:start_w + img_w]

    if background.shape[:2] != (img_h, img_w):
        # If the background is still smaller in any dimension, pad it
        pad_h = max(0, img_h - background.shape[0])
        pad_w = max(0, img_w - background.shape[1])

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        background = cv2.copyMakeBorder(
            background, top, bottom, left, right, 
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    return background

import numpy as np
from PIL import Image
import random

def add_background_image(img):
    if not background_list: 
        return img.astype(np.uint8)

    # Load and process background
    background_image = np.array(Image.open(random.choice(background_list)))
    background_image = resize_or_pad_background(background_image, img)

    # 1. Normalize Fingerprint to 0.0 - 1.0 float range
    # Assumption: img is 0 (ink) to 255 (white)
    fp_normalized = img.astype(float) / 255.0

    # 2. Handle Dimensions (Broadcasting)
    # If background is RGB (H,W,3) and fingerprint is Grayscale (H,W),
    # we need to add a dimension to fingerprint so numpy can multiply them.
    if len(background_image.shape) == 3 and len(fp_normalized.shape) == 2:
        fp_normalized = fp_normalized[..., np.newaxis]

    # 3. Define "Ink Intensity" (similar to your Alpha)
    # 1.0 = Pitch black ink (Standard Multiply)
    # 0.5 = Faded grey ink
    # We tweak the fp_normalized values to be closer to 1 (white) if we want less intensity.
    intensity = random.uniform(0.5, 1.0) 
    
    # Math: map the fingerprint range [0, 1] to [1-intensity, 1]
    # This ensures the ink never goes below a certain darkness, preventing pitch-black artifacts.
    # fp_blend_mask = fp_normalized * intensity + (1 - intensity)

    # 4. Perform the Multiply Blend
    # Formula: Background * FingerprintMask
    # result = background_image.astype(float) * fp_blend_mask

    fp_multiply_layer = 1.0 - ((1.0 - fp_normalized) * random.uniform(0.8, 1.0)) # Opacity control

    doc_float = background_image.astype(float) / 255.0
    
    # Ensure dimensions match for broadcasting
    if len(doc_float.shape) == 3:
        fp_multiply_layer = fp_multiply_layer[..., np.newaxis]

    # Apply Multiply
    result = doc_float * fp_multiply_layer

    # 4. Perform the Multiply Blend
    # Formula: Background * FingerprintMask
    # result = img.astype(float) * bg_blend_mask

    # 5. Clip and Convert back to uint8
    # return np.clip(result, 0, 255).astype(np.uint8)

    # result is currently 0.0 - 1.0. We must multiply by 255.
    return np.clip(result * 255, 0, 255).astype(np.uint8)
    

def vary_ridge_thickness(img):
    apply = np.random.uniform(0, 1)
    if apply <= 0.5:
        return img

    operator_size = np.random.randint(2, 5)
    kernel = np.ones((operator_size, operator_size), np.uint8)

    if np.random.uniform(0, 1) > 0.5:
        return cv2.dilate(img, kernel, iterations=1)

    return cv2.erode(img, kernel, iterations=1)

def gaussian_noise_speckle(img, noise_level):
    gaussian = np.random.normal(0, noise_level, img.shape)
    noisy_img = img + img * gaussian
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def parabolic_occlusion(img, thickness_range = (7, 15), opacity_range = (0.3,0.9)):
    line_color = 0 if random.uniform(0,1) < 0.5 else 255  

    # Defining parabola parameters
    ylim, xlim = img.shape
    signal     = random.choice([-1, 1])
    a          = signal * random.uniform(0.001, 0.009)
    b          = -a * random.uniform(0.1 * xlim, 0.9 * xlim)
    c          = random.uniform(0, 2 * xlim) if signal == 1 else random.uniform(-xlim, xlim)
    x          = np.linspace(0, xlim, 1000)
    y          = np.polyval([a, b, c], x)
    draw_pts   = (np.asarray([x, y]).T).astype(int)

    # Defining drawing parameters
    thickness = random.randint(*thickness_range)
    opacity   = random.uniform(*opacity_range)


    # Drawing image
    drawn     = np.ones_like(img)
    drawn     = cv2.polylines(drawn, [draw_pts], False, (line_color,line_color,line_color), thickness)

    out       = np.where(drawn == line_color, ((opacity) * drawn + (1 - opacity) * img), img).astype(np.uint8)
    
    
    return out


def linear_occlusion(img, thickness_range = (7, 15), opacity_range = (0.3, 0.9), n_lines_range = (3, 7)):
    n_lines = random.randint(*n_lines_range)
    out     = img.copy()

    for i in range(n_lines):
        line_color = 0 if random.uniform(0,1) < 0.5 else 255  

        # Defining line parameters
        xlim, ylim = img.shape

        xstart, xend = random.sample(range(0, xlim), 2)
        ystart, yend = random.sample(range(0, ylim), 2)

        walls      = {'left': (0, ystart), 'right': (xlim, yend), 'top': (xstart, 0), 'bottom': (xend, ylim)}
        start, end = random.sample(list(walls.items()), 2)

        # Defining drawing parameters
        thickness = random.randint(*thickness_range)
        opacity   = random.uniform(*opacity_range)

        # Drawing image

        drawn = (np.ones_like(img, dtype=np.uint8))
        drawn = cv2.line(drawn, start[1], end[1], (line_color,line_color,line_color), thickness)

        out       = np.where(drawn == line_color, ((opacity) * drawn + (1 - opacity) * out), out).astype(np.uint8)

    return out

# Modified generateSyntheticLatents.py with CLI toggles for synthetic transformations



# (Keep all function definitions as is, unchanged...)
# Only edits are: add CLI flags, update degradation_block, and update generate_synthetic_latent()

# Add these CLI toggles
TRANSFORMATION_FLAGS = {
    'occlusion':     ('--occlusion',      'Add parabolic and linear occlusion'),
    'thickness':     ('--thickness',      'Add thickness variation'),
    'illumination':  ('--illumination',   'Add gamma and downsampling'),
    'degradation':   ('--degradation',    'Add gaussian blur and noise'),
    'background':    ('--background',     'Add background image'),
}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("list", type=str, help="List with all fingerprint reference images (.tif, .png, .bmp, .wsq, .jpg)")
    parser.add_argument("n", type=int, help="Number of synthetic latents to generate per reference image")
    parser.add_argument("output", type=str, help="Output folder")
    parser.add_argument("background_path", type=str, help="Folder with background images")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of parallel workers (default: all CPUs)")

    for flag, desc in TRANSFORMATION_FLAGS.values():
        parser.add_argument(flag, action='store_true', help=desc)
    args = parser.parse_args()
    print("\nTransformations enabled:")
    if args.occlusion:
        print("- Parabolic and linear occlusion")
    if args.thickness:
        print("- Thickness variation")
    if args.illumination:
        print("- Gamma correction and downsampling")
    if args.degradation:
        print("- Gaussian blur and noise")
    if args.background:
        print("- Background image")
    print()

    return args

def generate_synthetic_latent(img, args):
    low_or_high = random.choice([0, 1])
    gamma_value = random.uniform(0.3, 0.7) if low_or_high == 0 else random.uniform(3.0, 4.0)
    noise_level = random.uniform(0.1, 0.3)
    blur_size = random.choice([num for num in range(3, 5) if num % 2 != 0])
    downsample = random.choice([num for num in range(2, 4)])

    out = img.copy()

    if args.occlusion:
        out = parabolic_occlusion(out)
        out = linear_occlusion(out)
    if args.thickness:
        out = vary_ridge_thickness(out)
    if args.illumination:
        out = gamma(out, value=gamma_value)
    if args.degradation:
        out = gaussian_blur(out, kernel_size=blur_size)
    # if args.illumination:
    #     out = downsampling(out, block_size=downsample)
    if args.degradation:
        out = gaussian_noise_speckle(out, noise_level=noise_level)
    if args.background:
        out = add_background_image(out)

    return out.astype(int)

def process_image(args_tuple):
    try:
        image_path, output_dir, args = args_tuple
        img = np.array(Image.open(image_path))
        basename = os.path.basename(image_path)

        for j in range(args.n):
            print(f"\nProcessing {basename} - synthetic {j}")
            out = generate_synthetic_latent(img, args)
            filename = os.path.join(output_dir, basename.replace('.png', f'_aug{j}.png'))
            cv2.imwrite(filename, out.astype(np.uint8))
    except Exception as e:
        print("Failed for image ", basename, "Error: ", e)

def read_image_list(image_list):
    with open(image_list, 'r') as fp:
        lines = fp.readlines()
    return [line.strip() for line in lines]

def create_output_dir(directory):
    try:
        os.mkdir(directory)
    except FileExistsError:
        print('Warning: Output folder already exists. Files may be overwritten.')

def main(args):
    images = read_image_list(args.list)
    create_output_dir(args.output)

    global background_list
    background_list = glob.glob(args.background_path + '/**/*.tif', recursive=True)

    task_args = [(image, args.output, args) for image in images]

    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(process_image, task_args), 1):
            if i % 10 == 0:
                print(f"Processed {i}/{len(images)} images")

if __name__ == '__main__':
    main(parse_args())
