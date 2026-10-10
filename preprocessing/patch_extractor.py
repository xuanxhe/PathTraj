import os
import json
import h5py
import glob
import argparse
import numpy as np
import pandas as pd
import scanpy as sc

from tqdm import tqdm
from pathlib import Path
from openslide import OpenSlide


def parse_args():
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument("--base-path", type=str, default="/home/andrewtal/SSD1/projects/PHST/patch_ee/data/HEST-1k", help="None")
    return parser.parse_args()


def process_st(index, base_path, target_size=224, is_spot_size=True):
    mark = 'spot' if is_spot_size else 'fix'
    output_file = '{}/patches_ext/patches_{}/{}.h5'.format(base_path, mark, index)
    os.makedirs(Path(output_file).parent, exist_ok=True)

    if os.path.exists(output_file):
        return
    
    h5ad_path = '{}/st/{}.h5ad'.format(base_path, index)
    slide_path = '{}/wsis/{}.tif'.format(base_path, index)
    metadata_path = '{}/metadata/{}.json'.format(base_path, index)

    slide = OpenSlide(slide_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
        
    mpp = metadata['pixel_size_um_estimated']

    if is_spot_size:
        try:
            spot_diameter = float(metadata['spot_diameter'])
        except:
            print('Sample dont have spot diameter.')
            return 0
            
        if spot_diameter not in [55.0, 100.0, 150.0]:
            return 0
            
        sd_pixel = round(spot_diameter / mpp)
    else:
        # 0.5um/pixel * 224 pixel size
        sd_pixel = round(0.5 * target_size / mpp)
    
    adata = sc.read_h5ad(h5ad_path)
    coords_center = adata.obsm['spatial']
    barcodes = adata.obs.index.tolist()
    
    coords_center = np.rint(coords_center - sd_pixel/2).astype(int)
    num_images = len(coords_center)
    
    with h5py.File(output_file, "w") as f:
        img_dset = f.create_dataset(
            "img",
            shape=(num_images, target_size, target_size, 3),
            dtype=np.uint8,
            compression=None
        )
        
        dt = h5py.string_dtype(encoding='utf-8')
        barcode_dset = f.create_dataset("barcode", (num_images, 1), dtype=dt)
        coord_dset = f.create_dataset("coords", shape=(num_images, 2), dtype=int)
        
        for i, coord in enumerate(coords_center):
            patch = slide.read_region(
                (coord[0], coord[1]),
                0,
                (sd_pixel, sd_pixel)
            ).convert('RGB').resize((target_size, target_size))
            patch_ar = np.array(patch)

            img_dset[i] = patch_ar
            barcode_dset[i] = [barcodes[i]]
            coord_dset[i] = coord

    return output_file



if __name__ == "__main__":
    args = parse_args()
    indexs = [Path(item).stem for item in glob.glob('{}/st/*.h5ad'.format(args.base_path))]

    for index in tqdm(indexs, total=len(indexs)):
        process_st(index, args.base_path, is_spot_size=True) # For Spot Size
        process_st(index, args.base_path, is_spot_size=False) # For 0.5um/pixel (fix)

