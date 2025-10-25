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

import scipy.ndimage as ndi
from skimage.draw import polygon
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
from shapely.geometry.polygon import Polygon
import skimage.morphology as morph


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

def add_background_image(img, background_img=None):
    if background_img is not None:
        background_image = background_img
    else:
        background_image = np.array(Image.open(random.choice(background_list)))

    # print('Background shape:', bg_shape)
    # print('Image shape:', img.shape)

    # if bg_shape != img.shape:
    #     background_image = center_crop(background_image, img.shape)

    background_image = resize_or_pad_background(background_image, img)

    alpha = random.uniform(0.2, 0.8)
    return (alpha * img + (1 - alpha) * background_image).astype(np.uint8)
    

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

def generate_synthetic_latent(img, background_path):
    low_or_high = random.choice([0, 1])
    gamma_value = random.uniform(0.3, 0.7) if low_or_high == 0 else random.uniform(3.0, 4.0)
    noise_level = random.uniform(0.1, 0.3)
    blur_size = random.choice([num for num in range(3, 5) if num % 2 != 0])
    downsample = random.choice([num for num in range(2, 4)])

    degradation_block = [
        partial(vary_ridge_thickness),
        partial(gamma, value=gamma_value),
        partial(gaussian_blur, kernel_size=blur_size),
        partial(gaussian_noise_speckle, noise_level=noise_level),
        partial(add_background_image)
    ]

    out = img.copy()

    for d in degradation_block:
        out = d(out)

    return out.astype(int)

def generate_synthetic_latent_and_mask(img, mask, background_path):
    low_or_high = random.choice([0, 1])
    gamma_value = random.uniform(0.3, 0.7) if low_or_high == 0 else random.uniform(3.0, 4.0)
    noise_level = random.uniform(0.1, 0.3)
    blur_size = random.choice([num for num in range(3, 5) if num % 2 != 0])
    downsample = random.choice([num for num in range(2, 4)])
    bg_img = np.array(Image.open(random.choice(background_list)))

    degradation_block = [
        partial(vary_ridge_thickness),
        partial(gamma, value=gamma_value),
        partial(gaussian_blur, kernel_size=blur_size),
        partial(gaussian_noise_speckle, noise_level=noise_level),
        partial(add_background_image, background_img=bg_img)
    ]

    mask_degradation_block = [
        partial(gamma, value=gamma_value),
        partial(gaussian_noise_speckle, noise_level=noise_level),
        partial(add_background_image, background_img=bg_img)
    ]

    out = img.copy()
    out_mask = mask.copy()

    for d in degradation_block:
        out = d(out)
    
    for d in mask_degradation_block:
        out_mask = d(out_mask)

    out_mask = np.where(out_mask > out_mask.mean(), 1, 0)


    return out.astype(int), out_mask.astype(int)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("list", type=str, help="List with all fingerprint reference images (.tif, .png, .bmp, .wsq, .jpg)")
    parser.add_argument("n", type=int, help="Number of synthetic latents to generate per reference image")
    parser.add_argument("output", type=str, help="Output folder")
    parser.add_argument("mask_out_dir", type=str, help="Output folder for latent masks")
    parser.add_argument("background_path", type=str, help="Folder with background images")
    parser.add_argument("mask_path", type=str, help="Folder with binary masks (same filenames as input images)")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of parallel workers (default: all CPUs)")
    return parser.parse_args()

def create_output_dir(directory):
    try:
        os.mkdir(directory)
    except FileExistsError:
        print('Warning: Output folder already exists. Files may be overwritten.')

def random_occlusions(img, max_squares=40, max_lines=5):
    img = np.ones_like(img, dtype=np.uint8)
    h, w = img.shape[:2]

    # Add random black squares
    for _ in range(random.randint(10, max_squares)):
        square_size = random.randint(16, 20)
        x = random.randint(0, w - square_size)
        y = random.randint(0, h - square_size)
        img[y:y+square_size, x:x+square_size] = 0  # Black square

    # Add random black lines
    for _ in range(random.randint(3, max_lines)):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        thickness = 5
        cv2.line(img, (x1, y1), (x2, y2), color=0, thickness=thickness)

    return img.astype(np.uint8)

def generate_mask(base_mask, area_range=(16000, 32000)):
    """
    Gera uma máscara binária com formato côncavo aleatório
    dentro da região base_mask (onde base_mask == 1).
    """
    h, w = base_mask.shape
    coords = np.argwhere(base_mask == 1)

    if len(coords) == 0:
        raise ValueError("Mask is empty.")

    # Escolhe pontos aleatórios da máscara
    num_points = np.random.randint(8, 16)

    invalidPolygon = True
    while (invalidPolygon):
        selected = coords[np.random.choice(len(coords), size=8, replace=False)]

        # Envolve os pontos com uma forma côncava usando morfologia
        temp_mask = np.zeros_like(base_mask, dtype=np.uint8)
        for y, x in selected:
            temp_mask[y, x] = 1

        height, width = base_mask.shape
        center = (width // 2, height // 2)
        radius = min(width, height) * random.uniform(*(0.2,0.5))
        
        angles = sorted([random.uniform(0, 2 * np.pi) for _ in range(6)])
        points = []

        for angle in angles:
            r = radius * random.uniform(0.8, 1.2)  # pequeno ruído no raio
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            points.append((x, y))

        polygon = np.array([points], dtype=np.int32)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, polygon, color=1)

        # Calcula a área do polígono
        area = cv2.contourArea(polygon[0])

        if area > area_range[0] and area < area_range[1]: # garante poligono com área minima de 128x128 pixels^2
            invalidPolygon = False

    # Interseção com a máscara original para manter dentro dos limites
    concave_mask = np.logical_and(mask, base_mask)

    return concave_mask.astype(np.uint8)


def process_image(args):
    try:
        image_path, output_dir, mask_out_dir, background_path, num_synthetic, mask_path = args
        img = np.array(Image.open(image_path))
        basename = os.path.basename(image_path)
        mask_file = os.path.join(mask_path, basename)
        mask = np.array(Image.open(mask_file).convert('L')) > 127  # binariza
        mask = mask.astype(np.uint8)

        for j in range(num_synthetic):
            # occ_mask = random_occlusions(mask)
            # img = np.where(new_mask == 0, img.max(), img)
            out, out_mask = generate_synthetic_latent_and_mask(img, mask, background_path)
            filename = os.path.join(output_dir, basename.replace('.png', f'_aug{j}.png'))
            cv2.imwrite(filename, out.astype(np.uint8))
            
            mask_filename = os.path.join(mask_out_dir, basename.replace('.png', f'_aug{j}.png'))
            cv2.imwrite(mask_filename, out_mask*255)
    except Exception as e:
        print("Failed for image ", basename, "Error: ", e)


def read_image_list(image_list):
    with open(image_list, 'r') as fp:
        lines = fp.readlines()
    
    return [line.strip() for line in lines]

def main(args):
    # Reading input args
    images = read_image_list(args.list)
    create_output_dir(args.output)
    create_output_dir(args.mask_out_dir)

    global background_list
    background_list = glob.glob(args.background_path + '/**/*.tif', recursive=True)
    # Prepare arguments for multiprocessing
    task_args = [(image, args.output, args.mask_out_dir, args.background_path, args.n, args.mask_path) for image in images]


    # Use multiprocessing to parallelize image processing
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(process_image, task_args), 1):
            if i % 10 == 0:
                print(f"Processed {i}/{len(images)} images")

if __name__ == '__main__':
    main(parse_args())
