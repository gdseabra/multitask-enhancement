import argparse
import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def process_mask(mask_path, output_dir):
    try:
        # Load image as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Unable to read image {mask_path}")
            return

        # --- MODIFICATION START ---
        
        # 1. Morphological Closing to fuse 8x8 blocks together
        # We use an Ellipse kernel slightly larger than the 8x8 block size
        # to ensure adjacent blocks connect before we blur them.
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_morph)

        # 2. Strong Gaussian Blur (The "Rounding" Step)
        # The kernel size is set to (31, 31). 
        # For 8x8 blocks, a blur radius ~4x the block size melts the squareness completely.
        blurred = cv2.GaussianBlur(closed, (31, 31), 0)

        # 3. Fixed Thresholding (The "Cut" Step)
        # We cut at 127 (middle gray). This creates the sharp binary edge 
        # based on the smooth gradient created by the blur.
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # --- MODIFICATION END ---

        # Save with same basename in output dir
        basename = os.path.basename(mask_path)
        out_path = os.path.join(output_dir, basename)
        cv2.imwrite(out_path, binary)
        print(f"Saved: {out_path}")

    except Exception as e:
        print(f"Error processing {mask_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Apply morphological post-processing to mask images in parallel.")
    parser.add_argument('--input_list', required=True, help='Path to a text file listing input mask image paths')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed masks')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (defaults to CPU count)')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Read mask paths
    with open(args.input_list, 'r') as f:
        mask_paths = [line.strip() for line in f if line.strip()]

    # Parallel processing
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        func = partial(process_mask, output_dir=args.output_dir)
        executor.map(func, mask_paths)

if __name__ == "__main__":
    main()