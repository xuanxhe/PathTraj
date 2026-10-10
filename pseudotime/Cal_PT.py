import argparse
import json
import os
import warnings
import h5py
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import stlearn as st
import scipy.stats as stats
from pathlib import Path
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import comb
from scipy.stats import kendalltau, pearsonr, spearmanr, wasserstein_distance
from shapely.geometry import MultiPolygon, Point, Polygon, shape as shapely_shape
from shapely.ops import unary_union
from sklearn.decomposition import PCA
from tqdm import tqdm

plt.switch_backend('Agg')
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12

# ==================== Parameter configuration ====================
#Sample Data Directory HEST-1K datasets
ROOT_DIR = "data"  
#Result Saving Directory：Store separately based on image features
SAVE_ROOT = "results/UNI"
MIN_SPOTS = 100
N_PCS = 30
HVG_N = None
RESOLUTION = 0.3
DIFFMAP = True

# ==================== Genetic feature pattern：HVGs or SVGs ====================
GENE_SELECTION_MODE = "HVG"  # "HVG" or "SVG"
SVG_TOP_N = 500              #  the number of genes
SVG_NEIGHBORS = 6            #  the number of neighbors used in calculating Moran's I

# ==================== Root node selection mode ====================
ROOT_SELECTION_MODE = "gene"  # Optional: "gene" (genetic feature) or "image" (image feature)
FEATURE_KEY = "UNI_features" #The features extracted from the pathological model
BARCODE_KEY = "barcode"
CELLVIT_DIRNAME = "cellvit_seg" 
CELLVIT_FILENAME_TEMPLATE = "{sample}_cellvit_seg.geojson"
IROOT_STRATEGY = 'no_tumor_else_min'

# ==================== Obtain the spatial coordinates of the spot ====================
def get_spatial_coords(adata, sample_name):

    if "pxl_col_in_fullres" in adata.obs.columns and "pxl_row_in_fullres" in adata.obs.columns:
        x_coords = adata.obs["pxl_col_in_fullres"].astype(float).values
        y_coords = adata.obs["pxl_row_in_fullres"].astype(float).values
        valid_mask = ~(np.isnan(x_coords) | np.isinf(x_coords) | (x_coords <= 0) |
                      np.isnan(y_coords) | np.isinf(y_coords) | (y_coords <= 0))
        return x_coords, y_coords, valid_mask
    
    if "spatial" in adata.obsm and adata.obsm["spatial"].shape[0] == adata.n_obs and adata.obsm["spatial"].shape[1] >= 2:
        x_coords = adata.obsm["spatial"][:, 0].astype(float)
        y_coords = adata.obsm["spatial"][:, 1].astype(float)
        valid_mask = ~(np.isnan(x_coords) | np.isinf(x_coords) | (x_coords <= 0) |
                      np.isnan(y_coords) | np.isinf(y_coords) | (y_coords <= 0))
        if valid_mask.sum() > 0:
            return x_coords, y_coords, valid_mask
        else:
            raise ValueError(f"[{sample_name}] adata.obsm['spatial'] no valid coordinates")
    
    raise KeyError(f"[{sample_name}] no space coordinates were found")

def get_spot_coordinates(adata, sample_name):
    x_coords, y_coords, _ = get_spatial_coords(adata, sample_name)
    return np.vstack([x_coords, y_coords]).T

# ====================  Verification of spot alignment between genes and images ==================== 
def verify_spot_alignment(adata_gene, adata_img, sample_name, save_dir):
    verification_results = {
        "sample": sample_name,
        "gene_spot_count": adata_gene.n_obs,
        "image_spot_count": adata_img.n_obs,
        "length_match": False,
        "barcode_set_match": False,
        "barcode_order_match": False,
        "spatial_coordinate_match": False,
        "alignment_status": "FAIL"
    }

    if adata_gene.n_obs != adata_img.n_obs:
        verification_results["alignment_status"] = "FAIL: Length mismatch"
        report_str = f"Inconsistent lengths: ST data {adata_gene.n_obs} Image data {adata_img.n_obs} \n"
        print(report_str)
        raise RuntimeError(f"\n{sample_name} spot alignment verification failed:{verification_results['alignment_status']}")
    else:
        verification_results["length_match"] = True
       
        gene_barcodes = set(adata_gene.obs_names.astype(str))
        img_barcodes = set(adata_img.obs_names.astype(str))
        
        if gene_barcodes == img_barcodes:
            verification_results["barcode_set_match"] = True

            common_barcodes = list(gene_barcodes) 
            img_barcode_to_idx = {bc: i for i, bc in enumerate(adata_img.obs_names.astype(str))}
            new_order = [img_barcode_to_idx[bc] for bc in adata_gene.obs_names.astype(str)]
            adata_img_reordered = adata_img[new_order].copy()

            if all(adata_gene.obs_names.astype(str) == adata_img_reordered.obs_names.astype(str)):
                verification_results["barcode_order_match"] = True
            else:
                verification_results["alignment_status"] = "FAIL: Barcode order mismatch after reordering"
                raise RuntimeError(f"\n{sample_name} Spot alignment verification failed:{verification_results['alignment_status']}")
            

            gene_coords = get_spot_coordinates(adata_gene, sample_name)
            img_coords = get_spot_coordinates(adata_img_reordered, sample_name)
            spatial_diff = np.abs(gene_coords - img_coords).max()
            if spatial_diff < 1e-3:
                verification_results["spatial_coordinate_match"] = True
            else:
                verification_results["alignment_status"] = "FAIL: Spatial coordinate mismatch"
                raise RuntimeError(f"\n{sample_name} spot alignment verification failed:{verification_results['alignment_status']}")

            verification_results["alignment_status"] = "PASS"
            pd.DataFrame([verification_results]).to_csv(
                os.path.join(save_dir, f"{sample_name}_spot_alignment_verification.csv"),
                index=False, encoding="utf-8-sig"
            )
            return adata_img_reordered
            
        else:
            only_gene = gene_barcodes - img_barcodes
            only_img = img_barcodes - gene_barcodes
            verification_results["alignment_status"] = "FAIL: Barcode set mismatch"
            report_str = f"barcode set inconsistency\n"
            if only_gene:
                report_str += f"Only in the barcode section of the genetic data:{list(only_gene)[:5]}...\n"
            if only_img:
                report_str += f"Only the barcodes in the image data:{list(only_img)[:5]}...\n"
            print(report_str)
            raise RuntimeError(f"\n{sample_name} spot alignment verification failed:{verification_results['alignment_status']}")


# ==================== Sample resolution acquisition function ====================
def get_sample_pixel_resolution(sample_name, sample_dir):
    metadata_path = os.path.join(sample_dir, "metadata", f"{sample_name}.json")
    if not os.path.exists(metadata_path):
        print(f"[{sample_name}] metadata file not found:{metadata_path}")
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
        if 'pixel_size_um_estimated' in metadata:
            pixel_resolution = float(metadata['pixel_size_um_estimated'])
            print(f"[{sample_name}] resolution:{pixel_resolution:.6f} μm/pixel")
            return pixel_resolution
        else:
            print(f"[{sample_name}] metadata has no 'pixel_size_um_estimated' ")
            return None
    except Exception as e:
        print(f"[{sample_name}] Failed to read metadata:{str(e)[:50]}")
        return None

# ==================== Read the spot radius ====================
def get_real_spot_pixel_radius(sample_name, sample_dir):

    pixel_resolution = get_sample_pixel_resolution(sample_name, sample_dir)
    if pixel_resolution is None:
        print(f"[{sample_name}] Unable to obtain resolution")
        return None
    
    metadata_path = os.path.join(sample_dir, "metadata", f"{sample_name}.json")
    spot_physical_diam = None
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if 'spot_diameter' in metadata:
                spot_physical_diam = float(metadata['spot_diameter'])
                print(f"[{sample_name}] Spot physical diameter:{spot_physical_diam:.2f} μm")
        except Exception as e:
            print(f"[{sample_name}] spot has no physical diameter")
    
    if spot_physical_diam is None:
        warnings.warn(f"[{sample_name}] metadata missing spot_diameter")
        return None
    spot_pixel_diam = spot_physical_diam / pixel_resolution
    spot_pixel_radius = spot_pixel_diam / 2.0  
    print(f"[{sample_name}] actual Spot pixel radius:{spot_pixel_radius:.2f} pixel")
    return spot_pixel_radius

# ==================== Read the outline of the tissue ====================
def load_polygon_for_sample(sample_dir, sample_name):
    geojson_path = os.path.join(sample_dir, "tissue_seg", f"{sample_name}_contours.geojson")
    if not os.path.exists(geojson_path):
        return None, geojson_path
    try:
        gdf = gpd.read_file(geojson_path)
        polys = [geom for geom in gdf.geometry if geom is not None]
        if len(polys) == 0:
            return None, geojson_path
        if len(polys) == 1:
            return polys[0], geojson_path
        return unary_union(polys), geojson_path  
    except Exception as e:
        try:
            with open(geojson_path, 'r', encoding='utf-8') as fh:
                gj = json.load(fh)
            feats = gj.get('features', None)
            if feats is None:
                geom = gj.get('geometry', gj)
                shp = shapely_shape(geom)
                return shp, geojson_path
            polys = []
            for f in feats:
                g = f.get('geometry', None)
                if g is None:
                    continue
                polys.append(shapely_shape(g))
            if len(polys) == 0:
                return None, geojson_path
            if len(polys) == 1:
                return polys[0], geojson_path
            return unary_union(polys), geojson_path  
        except Exception as e2:
            print(f"Unable to read geojson：{e} / {e2}")
            return None, geojson_path

# ==================== Read the cell segmentation data ====================
def load_cellvit_polygons(sample_dir, sample_name):
    fname = os.path.join(sample_dir, CELLVIT_DIRNAME, CELLVIT_FILENAME_TEMPLATE.format(sample=sample_name))
    if not os.path.exists(fname):
        return None, None, fname
    try:
        with open(fname, 'r', encoding='utf-8') as fh:
            gj = json.load(fh)
    except Exception as e:
        warnings.warn(f"Unable to read cellvit geojson: {fname} - {e}")
        return None, None, fname

    if isinstance(gj, list):
        features = gj
    elif isinstance(gj, dict):
        features = gj.get('features', None)
        if features is None:
            geom = gj.get('geometry', gj)
            features = [{'geometry': geom, 'properties': gj.get('properties', {})}]
    else:
        warnings.warn(f"Unknown JSON structure: {fname}")
        return None, None, fname

    cell_polygons = []
    cell_types = []
    for feat in features:
        if not isinstance(feat, dict):
            continue
        geom = feat.get('geometry', None)
        props = feat.get('properties', {})
        cls = None
        if 'classification' in props:
            cls = props['classification'].get('name', None) if isinstance(props['classification'], dict) else props['classification']
        if cls is None:
            cls = props.get('label', props.get('name', None))
        if geom is None:
            continue
        try:
            shp = shapely_shape(geom)
            if shp is None:
                continue
            if isinstance(shp, MultiPolygon):
                for p in shp.geoms:
                    cell_polygons.append(p)
                    cell_types.append(cls if cls is not None else "Unknown")
            else:
                cell_polygons.append(shp)
                cell_types.append(cls if cls is not None else "Unknown")
        except Exception:
            continue
    n_features = len(features)
    n_polygons = len(cell_polygons)    
    print(f"geojson features count: {n_features}")
    print(f"atomic polygon count: {n_polygons}")
    return cell_polygons, cell_types, fname

# ==================== Calculation of tumor cell proportion ====================
def compute_spot_cell_counts_from_seg_fast_with_composition(adata, cell_polygons, cell_types, sample_name, sample_dir):
    x_coords, y_coords, valid_mask = get_spatial_coords(adata, sample_name)
    n_total_spots = len(x_coords)    
    if not valid_mask.any():
        empty_composition = pd.DataFrame(columns=['spot_index', 'spot_barcode', 'cell_id', 'cell_type', 
                                                  'overlap_area', 'cell_centroid_x', 'cell_centroid_y'])
        return (np.zeros(n_total_spots, int), np.zeros(n_total_spots, int), 
                np.zeros(n_total_spots, float), empty_composition)

    spot_r = get_real_spot_pixel_radius(sample_name, sample_dir)
    if spot_r is None:
        print(f"[{sample_name}] Unable to obtain the spot radius")
        empty_composition = pd.DataFrame(columns=['spot_index', 'spot_barcode', 'cell_id', 'cell_type', 
                                                  'overlap_area', 'cell_centroid_x', 'cell_centroid_y'])
        return (np.zeros(n_total_spots, int), np.zeros(n_total_spots, int), 
                np.zeros(n_total_spots, float), empty_composition)

    spot_geoms = []
    spot_original_idx = []
    spot_barcodes = []
    for idx in range(n_total_spots):
        if not valid_mask[idx]:
            continue
        x, y = x_coords[idx], y_coords[idx]
        spot_geoms.append(Point(x, y).buffer(spot_r))
        spot_original_idx.append(idx)
        spot_barcodes.append(adata.obs_names[idx])
    
    if len(spot_original_idx) == 0:
        empty_composition = pd.DataFrame(columns=['spot_index', 'spot_barcode', 'cell_id', 'cell_type', 
                                                  'overlap_area', 'cell_centroid_x', 'cell_centroid_y'])
        return (np.zeros(n_total_spots, int), np.zeros(n_total_spots, int), 
                np.zeros(n_total_spots, float), empty_composition)
    
    spots_gdf = gpd.GeoDataFrame({
        'spot_original_idx': spot_original_idx,
        'spot_barcode': spot_barcodes
    }, geometry=spot_geoms, crs="EPSG:4326").rename_geometry('geometry_spot')

    cell_ids = np.arange(len(cell_polygons))
    valid_cell_data = []
    rejected_records = []
    for cid, geom, ctype in zip(cell_ids, cell_polygons, cell_types):
        reason = None
        if geom is None:
            reason = "no_geom"
        elif ctype is None:
            reason = "no_type"
        else:
            if not isinstance(geom, (Polygon, MultiPolygon)):
                reason = "not_polygon"
            else:
                try:
                    if not geom.is_valid or geom.area == 0:
                        reason = "invalid_geometry"
                except Exception:
                    reason = "geom_check_error"
        if reason is not None:
            try:
                cx, cy = (geom.centroid.x, geom.centroid.y) if geom is not None else (np.nan, np.nan)
            except Exception:
                cx, cy = (np.nan, np.nan)
            rejected_records.append({"orig_cell_id": int(cid), "reason": reason, 
                                    "centroid_x": cx, "centroid_y": cy, 
                                    "cell_type": str(ctype) if ctype is not None else None})
            continue
        valid_cell_data.append({
            'cell_id': int(cid), 
            'cell_type': str(ctype).strip().lower(), 
            'geometry_cell': geom,
            'cell_centroid_x': geom.centroid.x,
            'cell_centroid_y': geom.centroid.y
        })

    cells_geoms = [d['geometry_cell'] for d in valid_cell_data]
    cells_gdf = gpd.GeoDataFrame(valid_cell_data, geometry=cells_geoms, 
                                crs="EPSG:4326").rename_geometry('temp_cell_geom')
    print(f"[{sample_name}] effective cell count:{len(cells_gdf)}")

    sample_save_dir = os.path.join(SAVE_ROOT, sample_name)
    os.makedirs(sample_save_dir, exist_ok=True)
    if len(rejected_records) > 0:
        df_rej = pd.DataFrame(rejected_records)
        rej_path = os.path.join(sample_save_dir, f"{sample_name}_filtered_cells.csv")
        df_rej.to_csv(rej_path, index=False, encoding="utf-8-sig")

    # Spatial connection
    try:
        join_res = gpd.sjoin(spots_gdf, cells_gdf, how='inner', predicate='intersects', 
                            lsuffix='_spot', rsuffix='_cell')
        if 'temp_cell_geom' in join_res.columns:
            join_res = join_res.drop(columns=['temp_cell_geom'])
    except Exception as e:
        raise RuntimeError(f"[{sample_name}] Space connection failed:{e}")

    # Calculate the overlapping area
    def calc_overlap_area(row):
        try:
            spot_geom = row['geometry_spot']
            cell_geom = row['geometry_cell']
            if not (isinstance(spot_geom, (Polygon, MultiPolygon)) and spot_geom.is_valid):
                return 0.0
            if not (isinstance(cell_geom, (Polygon, MultiPolygon)) and cell_geom.is_valid):
                return 0.0
            inter = spot_geom.intersection(cell_geom)
            return max(inter.area, 0.0) if inter and inter.is_valid else 0.0
        except Exception:
            return 0.0

    join_res['overlap_area'] = join_res.apply(calc_overlap_area, axis=1)
    join_res = join_res[join_res['overlap_area'] > 1e-6].copy()
    if len(join_res) == 0:
        empty_composition = pd.DataFrame(columns=['spot_index', 'spot_barcode', 'cell_id', 'cell_type', 
                                                  'overlap_area', 'cell_centroid_x', 'cell_centroid_y'])
        return (np.zeros(n_total_spots, int), np.zeros(n_total_spots, int), 
                np.zeros(n_total_spots, float), empty_composition)

    # 1cell->1spot（Select the spot with the largest overlapping area）
    join_res_sorted = join_res.sort_values(by=['cell_id', 'overlap_area'], ascending=[True, False])
    join_res_unique = join_res_sorted.drop_duplicates(subset='cell_id', keep='first')
    print(f"[{sample_name}] After removing duplicates:{len(join_res_unique)}")

    # Construct a DataFrame based on cell composition
    cell_composition_df = pd.DataFrame({
        'spot_index': join_res_unique['spot_original_idx'].values,
        'spot_barcode': join_res_unique['spot_barcode'].values,
        'cell_id': join_res_unique['cell_id'].values,
        'cell_type': join_res_unique['cell_type'].values,
        'overlap_area': join_res_unique['overlap_area'].values,
        'cell_centroid_x': join_res_unique['cell_centroid_x'].values,
        'cell_centroid_y': join_res_unique['cell_centroid_y'].values
    })

    spot_coords = {row['spot_original_idx']: (x_coords[row['spot_original_idx']], y_coords[row['spot_original_idx']]) 
                   for _, row in spots_gdf.iterrows()}
    cell_composition_df['spot_center_x'] = cell_composition_df['spot_index'].map(lambda x: spot_coords[x][0])
    cell_composition_df['spot_center_y'] = cell_composition_df['spot_index'].map(lambda x: spot_coords[x][1])

    total_count = np.zeros(n_total_spots, int)
    tumor_count = np.zeros(n_total_spots, int)

    spot_cell_stats = join_res_unique.groupby('spot_original_idx')['cell_id'].nunique()
    for spot_idx, count in spot_cell_stats.items():
        total_count[spot_idx] = int(count)
    
    # Statistics of Tumor Cells
    tumor_mask = join_res_unique['cell_type'].str.startswith('neoplastic', na=False)
    if tumor_mask.any():
        tumor_cells_df = join_res_unique[tumor_mask]
        spot_tumor_stats = tumor_cells_df.groupby('spot_original_idx')['cell_id'].nunique()
        for spot_idx, count in spot_tumor_stats.items():
            tumor_count[spot_idx] = int(count)
    else:
        print(f"[{sample_name}] No tumor cells were detected")

    # Calculate the proportion of tumors
    tumor_ratio = np.zeros(n_total_spots, float)
    for i in range(n_total_spots):
        if total_count[i] > 0:
            tumor_ratio[i] = tumor_count[i] / total_count[i]
        else:
            tumor_ratio[i] = 0.0
    
    return tumor_count, total_count, tumor_ratio, cell_composition_df

# ==================== Similarity indicator calculation function ====================
def compute_correlation_metrics(gene_ptime, img_ptime):
    """Correlation coefficient type indicators"""
    valid_mask = np.isfinite(gene_ptime) & np.isfinite(img_ptime)
    x = gene_ptime[valid_mask]
    y = img_ptime[valid_mask]
    n = len(x)
    if n < 3:
        print(f"Insufficient valid data points")
        return {k: np.nan for k in ['pearson_r', 'pearson_p', 'kendall_tau', 'kendall_p', 'spearman_r', 'spearman_p', 'distance_corr']}
    
    # Pearson
    pearson_r, pearson_p = pearsonr(x, y)
    # Kendall tau
    kendall_tau, kendall_p = kendalltau(x, y)
    # Spearman
    spearman_r, spearman_p = spearmanr(x, y)
    # Distance Correlation
    def distance_correlation(a, b):
        def _centered_distance_matrix(x):
            n = len(x)
            dist = cdist(x.reshape(-1, 1), x.reshape(-1, 1), 'euclidean')
            row_mean = dist.mean(axis=1, keepdims=True)
            col_mean = dist.mean(axis=0, keepdims=True)
            total_mean = dist.mean()
            return dist - row_mean - col_mean + total_mean
        dx = _centered_distance_matrix(x)
        dy = _centered_distance_matrix(y)
        cov_dxy = (dx * dy).sum() / (n ** 2)
        cov_dx = (dx ** 2).sum() / (n ** 2)
        cov_dy = (dy ** 2).sum() / (n ** 2)
        return np.sqrt(cov_dxy / np.sqrt(cov_dx * cov_dy)) if cov_dx > 0 and cov_dy > 0 else 0.0
    distance_corr = distance_correlation(x, y)

    return {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'kendall_tau': kendall_tau, 'kendall_p': kendall_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'distance_corr': distance_corr
    }

def compute_ranking_metrics(gene_ptime, img_ptime):
    """Sorting consistency index"""
    valid_mask = np.isfinite(gene_ptime) & np.isfinite(img_ptime)
    x = gene_ptime[valid_mask]
    y = img_ptime[valid_mask]
    n = len(x)
    if n < 2:
        print(f"Insufficient valid data points")
        return {k: np.nan for k in ['kendall_tau_b', 'kendall_tau_b_p', 'tau_distance', 
                                   'concordant_pairs', 'discordant_pairs']}

    rank_x = stats.rankdata(x, method='average')
    rank_y = stats.rankdata(y, method='average')
    
    # Kendall tau-b
    def kendall_tau_b(r1, r2):
        n = len(r1)
        concordant = 0
        discordant = 0
        ties_x = 0
        ties_y = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = r1[i] - r1[j]
                dy = r2[i] - r2[j]
                if dx * dy > 0:
                    concordant += 1
                elif dx * dy < 0:
                    discordant += 1
                else:
                    if dx == 0:
                        ties_x += 1
                    if dy == 0:
                        ties_y += 1
        numerator = concordant - discordant
        denominator = np.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
        tau_b = numerator / denominator if denominator != 0 else 0.0
        se = np.sqrt((n*(n-1)*(2*n+5) - ties_x*(2*n+5) - ties_y*(2*n+5) + 4*ties_x*ties_y/n) / (18*(n-1)*n**2))
        z = tau_b / se if se != 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(np.abs(z)))
        return tau_b, p_val, concordant, discordant
    
    kendall_tau_b_val, kendall_tau_b_p, concordant, discordant = kendall_tau_b(rank_x, rank_y)
    # Tau distance
    tau_distance = 0.5 * (1 - (concordant - discordant) / comb(n, 2)) if n >=2 else np.nan
    
    return {
        'kendall_tau_b': kendall_tau_b_val, 'kendall_tau_b_p': kendall_tau_b_p,
        'tau_distance': tau_distance,
        'concordant_pairs': concordant,
        'discordant_pairs': discordant
    }

def compute_distribution_metrics(gene_ptime, img_ptime):
    valid_mask = np.isfinite(gene_ptime) & np.isfinite(img_ptime)
    x = gene_ptime[valid_mask]
    y = img_ptime[valid_mask]
    if len(x) < 2 or len(y) < 2:
        print(f"Insufficient valid data points")
        return {'wasserstein_distance': np.nan}
    
    x_norm = (x - x.mean()) / (x.std() + 1e-8)
    y_norm = (y - y.mean()) / (y.std() + 1e-8)
    wasserstein_dist = wasserstein_distance(x_norm, y_norm)
    return {'wasserstein_distance': wasserstein_dist}

def compute_cosine_similarity(gene_ptime, img_ptime):
    valid_mask = np.isfinite(gene_ptime) & np.isfinite(img_ptime)
    x = gene_ptime[valid_mask]
    y = img_ptime[valid_mask]
    n = len(x)
    if n < 2:
        print(f"Insufficient valid data points")
        return np.nan
    
    x_norm = (x - x.mean()) / (x.std() + 1e-8)
    y_norm = (y - y.mean()) / (y.std() + 1e-8)
    dot_product = np.dot(x_norm, y_norm)
    x_norm_l2 = np.linalg.norm(x_norm)
    y_norm_l2 = np.linalg.norm(y_norm)
    
    return dot_product / (x_norm_l2 * y_norm_l2) if x_norm_l2 > 0 and y_norm_l2 > 0 else 0.0

# ==================== Visualization function for the proportion of spot tumors ====================
def plot_tumor_ratio_spatial(adata_before, adata_after, sample_name, save_dir, sample_dir=None):

    if 'tumor_ratio' not in adata_before.obs.columns:
        print(f"{sample_name} no tumor_ratio")
        return

    if 'cell_count' not in adata_before.obs.columns:
        print(f"{sample_name} no cell_count")
        return

    if sample_dir is None:
        sample_dir = os.path.join(ROOT_DIR, sample_name)

    poly, poly_path = load_polygon_for_sample(sample_dir, sample_name)
    has_tissue_contour = poly is not None

    kept_spots = set(adata_after.obs_names)
    mask_kept = adata_before.obs_names.isin(kept_spots)

    x_coords, y_coords, _ = get_spatial_coords(adata_before, sample_name)
    tumor_ratio = adata_before.obs['tumor_ratio'].astype(float).values

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])

    if mask_kept.any():
        sc = ax.scatter(
            x_coords[mask_kept],
            y_coords[mask_kept],
            c=tumor_ratio[mask_kept],
            s=25, cmap='Reds',
            vmin=0, vmax=1,
            edgecolors='k', linewidths=0.3,
            alpha=0.9, zorder=3
        )
    else:
        sc = None
        print(f"{sample_name}: unreserved spot")

    if has_tissue_contour:
        def plot_polygon(ax_obj, geom, color='black', linewidth=2.0, alpha=0.7):
            if isinstance(geom, Polygon):
                x_poly, y_poly = geom.exterior.xy
                ax_obj.plot(x_poly, y_poly, color=color, linewidth=linewidth,
                            alpha=alpha, zorder=5)
                for interior in geom.interiors:
                    xi, yi = interior.xy
                    ax_obj.plot(xi, yi, color=color, linewidth=linewidth-0.5,
                                alpha=alpha, linestyle='--', zorder=5)
            elif isinstance(geom, MultiPolygon):
                for sub_geom in geom.geoms:
                    plot_polygon(ax_obj, sub_geom, color, linewidth, alpha)

        plot_polygon(ax, poly)
        print(f"Organizational outline has been added: {poly_path}")

    ax.invert_yaxis()
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.tick_params(labelsize=11)
    ax.legend(loc='upper right', fontsize=28)

    if sc is not None:
        cbar_ax = fig.add_axes([0.82, 0.1, 0.03, 0.8])
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Tumor Ratio", fontsize=28)
        cbar.ax.tick_params(labelsize=28, width=1.5, length=4)

    save_path = os.path.join(save_dir, f"{sample_name}_tumor_ratio_spatial.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"The tumor proportion space graph has been saved:{save_path}")

# ==================== Pseudotime visualization function ====================
def plot_gene_pseudotime_distribution(gene_ptime, sample_name, save_dir):
    gene_ptime_clean = gene_ptime[np.isfinite(gene_ptime)]
    
    print(f"Number of effective values of gene pseudotime:{len(gene_ptime_clean)}/{len(gene_ptime)}")
    
    if len(gene_ptime_clean) == 0:
        print("There is no valid pseudotime data available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(gene_ptime_clean, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Pseudotime', fontsize=28)
    ax.set_ylabel('Frequency', fontsize=28)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{sample_name}_gene_pseudotime_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"The pseudotime distribution map of genes has been saved:{save_path}")

def plot_image_pseudotime_distribution(img_ptime, sample_name, save_dir):
    img_ptime_clean = img_ptime[np.isfinite(img_ptime)]
    
    print(f"Number of effective values of image pseudotime:{len(img_ptime_clean)}/{len(img_ptime)}")
    
    if len(img_ptime_clean) == 0:
        print("There is no valid pseudotime data available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.hist(img_ptime_clean, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.set_xlabel('Pseudotime', fontsize=28)
    ax.set_ylabel('Frequency', fontsize=28)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{sample_name}_image_pseudotime_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"The pseudotime distribution map of image has been saved: {save_path}")

def plot_and_save(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def save_spatial_with_root(adata_for_plot, color_key, root_coord, save_path, sample_name):
    fig_width = 10
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    sc.pl.spatial(
        adata_for_plot,
        color=[color_key],
        img_key="downscaled_fullres",  
        size=1.3,
        ncols=1,
        show=False,
        cmap="viridis",
        title='', 
        ax=ax
    )

    children = list(ax.get_children())
    cbar_to_remove = None
    for child in children:
        if hasattr(child, 'colorbar') and child.colorbar is not None:
            cbar_to_remove = child.colorbar
            break
        elif isinstance(child, plt.Axes) and child.get_label() == '<colorbar>':
            cbar_to_remove = child
            break
    if cbar_to_remove is not None:
        cbar_to_remove.remove()
           

    x_coords, y_coords, _ = get_spatial_coords(adata_for_plot, sample_name)
        
    if 'iroot' in adata_for_plot.uns:
        root_spot_name = adata_for_plot.obs_names[adata_for_plot.uns['iroot']]
        root_index = list(adata_for_plot.obs_names).index(root_spot_name)
    else:
        dists = np.sqrt((x_coords - root_coord[0])**2 + (y_coords - root_coord[1])**2)
        root_index = np.argmin(dists)
        
    sf = adata_for_plot.uns['spatial']['ST']['scalefactors']
    scale = sf['tissue_downscaled_fullres_scalef']
    spot_diam_fullres = sf['spot_diameter_fullres']
    spot_diam_plot = spot_diam_fullres * scale
    marker_size = (spot_diam_plot * 1.8) ** 2
            
    x_full, y_full = x_coords[root_index], y_coords[root_index]
    x_plot = x_full * scale
    y_plot = y_full * scale
   
    ax.scatter([x_plot], [y_plot],
            color='white',
            s=marker_size * 1.5,  
            marker='o',
            edgecolor='white',
            linewidths=1.5,
            zorder=9,  
            label='_nolegend_')  
    ax.scatter([x_plot], [y_plot],
            color='red',
            s=marker_size,
            marker='o',
            edgecolor='white',
            linewidths=1.5,
            zorder=10,
            label='Root')

    ax.legend(loc='upper right', frameon=True, framealpha=0.8, fontsize=28)
    ax_pos = ax.get_position()
    new_ax_width = 0.7  
    new_ax_left = 0.1   
    new_ax_bottom = 0.1  
    new_ax_height = 0.8  
    ax.set_position([new_ax_left, new_ax_bottom, new_ax_width, new_ax_height])
      
    cbar_left = new_ax_left + new_ax_width + 0.02  
    cbar_width = 0.03 
    cbar_bottom = new_ax_bottom  
    cbar_height = new_ax_height  
        
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        
    vals = adata_for_plot.obs[color_key].astype(float).values
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
     
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(color_key.replace('_', ' ').title(), fontsize=28)
    cbar.ax.tick_params(labelsize=28)
      
    ax.set_xlabel('X coordinate', fontsize=28)
    ax.set_ylabel('Y coordinate', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ==================== Organizational filtering function ====================
def filter_spots_by_tissue(adata, sample, sample_dir, save_dir):
    original_spots = adata.n_obs

    poly, poly_path = load_polygon_for_sample(sample_dir, sample)
    if poly is None:
        print(f"{poly_path} no organization outline file was found.")
        return adata, np.ones(adata.n_obs, dtype=bool), original_spots, original_spots

    x_coords, y_coords, valid_mask = get_spatial_coords(adata, sample)
    if not np.any(valid_mask):
        print("Without valid coordinates, organization filtering cannot be performed")
        return adata, valid_mask, original_spots, original_spots
    
    spot_r = get_real_spot_pixel_radius(sample, sample_dir)

    inside_mask = np.zeros(len(x_coords), dtype=bool)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if not valid_mask[i]:
            inside_mask[i] = False
            continue
            
        try:
            spot_circle = Point(x, y).buffer(spot_r)
            center_inside = poly.contains(Point(x, y))
            intersection = poly.intersection(spot_circle)
            area_inside = intersection.area / spot_circle.area if spot_circle.area > 0 else 0
            inside_mask[i] = center_inside or (area_inside > 0.7)
            
        except Exception as e:
            print(f"Error occurred while processing coordinates ({x}, {y}): {e} → Filtering")
            inside_mask[i] = False
    
    n_total = len(inside_mask)
    n_inside = int(inside_mask.sum())
    n_filtered = n_total - n_inside
    retention_rate = n_inside / n_total if n_total > 0 else 0
    print(f"tissue filter：Total spots{n_total} Reserved spots{n_inside} Filter spots{n_filtered}")
    

    try:
        fig, ax = plt.subplots(figsize=(8, 8))

        def _plot_polygon(ax_obj, geom):
            if isinstance(geom, Polygon):
                x_poly, y_poly = geom.exterior.xy
                ax_obj.plot(x_poly, y_poly, color='red', linewidth=1.6)
                for interior in geom.interiors:
                    xi, yi = interior.xy
                    ax_obj.plot(xi, yi, color='red', linewidth=0.8, alpha=0.7)
            elif isinstance(geom, MultiPolygon):
                for sub in geom.geoms:
                    _plot_polygon(ax_obj, sub)
        _plot_polygon(ax, poly)

        ax.scatter(
            x_coords[inside_mask & valid_mask], 
            y_coords[inside_mask & valid_mask], 
            s=8, c="blue", label=f"Inside Tissue ({n_inside} spots)"
        )
        ax.scatter(
            x_coords[~inside_mask & valid_mask], 
            y_coords[~inside_mask & valid_mask], 
            s=6, c="gray", alpha=0.5, label=f"Outside Tissue ({n_filtered} spots)"
        )
        ax.scatter(
            x_coords[~valid_mask], 
            y_coords[~valid_mask], 
            s=4, c="black", alpha=0.3, label=f"Invalid Coords ({len(x_coords)-valid_mask.sum()} spots)"
        )
        
        ax.invert_yaxis()
        ax.legend(fontsize=12)
        plot_and_save(fig, os.path.join(save_dir, f"{sample}_tissue_filter_check.png"))

    except Exception as e_vis:
        print(f"Failed to generate the organizational filtering result graph:{e_vis}")

    filtered_adata = adata[inside_mask].copy()
    return filtered_adata, inside_mask, original_spots, filtered_adata.n_obs

# ==================== Cell count filtering function ====================
def filter_spots_by_cell_count_enhanced(adata, sample_name=""):
    if 'cell_count' not in adata.obs.columns:
        mask = np.ones(adata.n_obs, dtype=bool)
        return adata.copy(), mask, None, 0
    cell_counts = adata.obs['cell_count'].astype(int).values

    mask_keep = (cell_counts > 0)
    n_total = len(mask_keep)
    n_kept = int(mask_keep.sum())
    n_filtered = n_total - n_kept
    
    filtered_adata = adata[mask_keep].copy()
    final_threshold = 1  
    return filtered_adata, mask_keep, cell_counts, final_threshold

# ==================== The function for checking the validity of pseudotime ====================
def check_pseudotime_validity_strict(ptime_array, sample_name, ptime_type):
    nan_count = np.sum(np.isnan(ptime_array))
    inf_count = np.sum(np.isinf(ptime_array))
    total_count = len(ptime_array)
    
    print(f"[{sample_name}] {ptime_type}Pseudotime validity check:")
    if nan_count > 0 or inf_count > 0:
        print(f"Discovering invalid values:NaN {nan_count} INF {inf_count} ")
        return False, nan_count, inf_count
    
    print(f"All pseudotime values are valid")
    return True, nan_count, inf_count

# ==================== Calculate the SVG gene ====================
def compute_morans_I(adata, n_neighbors=6):
    
    coords = adata.obsm['spatial']
    n_spots = coords.shape[0]

    tree = cKDTree(coords)
    dist, idx = tree.query(coords, k=n_neighbors + 1)
    idx = idx[:, 1:]

    rows = np.repeat(np.arange(n_spots), n_neighbors)
    cols = idx.ravel()
    data = np.ones(len(rows), dtype=np.float64)
    W = csr_matrix((data, (rows, cols)), shape=(n_spots, n_spots))
    W = W + W.T
    W.data[:] = 1.0
    
    S0 = W.sum()
    n = n_spots
    
    # Obtain the expression matrix
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)
    
    # Calculate Moran's I 
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    
    ss = np.sum(X_centered ** 2, axis=0)
    ss[ss == 0] = 1e-10
    
    WX = W.dot(X_centered)
    numerator = np.sum(X_centered * WX, axis=0)
    
    morans_I = (n / S0) * (numerator / ss)
    
    result = pd.DataFrame({'morans_I': morans_I}, index=adata.var_names)
    result = result.sort_values('morans_I', ascending=False)
    
    return result

def get_top_svg_genes_for_sample(adata, sample_name, n_neighbors=6, top_n=500):
    # Calculate the top SVG genes for individual samples based on the Moran's I value
    moran_df = compute_morans_I(adata, n_neighbors=n_neighbors)
    svg_top = moran_df.head(top_n).copy()
    svg_genes = svg_top.index.tolist()
    print(f"Found {len(svg_genes)} SVG genes")
    return svg_genes, svg_top

# ==================== Calculate the proportion of connective cells ====================
def compute_connective_ratios(adata_gene, cell_composition_df):
    if cell_composition_df is None or cell_composition_df.empty:
        return None
    spot_barcodes = adata_gene.obs_names.astype(str).tolist()
    
    spot_total = cell_composition_df.groupby('spot_barcode')['cell_id'].nunique().reset_index()
    spot_total.columns = ['spot_barcode', 'total_cells']
    
    connective_filtered = cell_composition_df[
        cell_composition_df['cell_type'].str.contains('connective', case=False, na=False)
    ]
    spot_connective = connective_filtered.groupby('spot_barcode')['cell_id'].nunique().reset_index()
    spot_connective.columns = ['spot_barcode', 'connective_count']
    
    spot_ratio = spot_total.merge(spot_connective, on='spot_barcode', how='left')
    spot_ratio['connective_count'] = spot_ratio['connective_count'].fillna(0)
    spot_ratio['connective_ratio'] = spot_ratio['connective_count'] / spot_ratio['total_cells']
    
    barcode_to_ratio = dict(zip(spot_ratio['spot_barcode'], spot_ratio['connective_ratio']))
    connective_ratios = np.array([barcode_to_ratio.get(bc, 0.0) for bc in spot_barcodes])
    
    return connective_ratios

# ==================== Select the root node based on the gene expression characteristics ====================
def select_root_from_gene_features(adata_gene, cell_composition_df, sample_name, resolution=RESOLUTION, diffmap=DIFFMAP):
    
    leiden_clusters = adata_gene.obs['leiden'].unique()

    # Calculate the proportion of connectives for each spot
    connective_ratios = compute_connective_ratios(adata_gene, cell_composition_df)

    # Calculate the comprehensive indicators for each cluster
    cluster_stats = {}
    for cluster_id in leiden_clusters:
        cluster_mask = adata_gene.obs['leiden'] == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        # Calculate the average tumor proportion
        tumor_ratios = adata_gene.obs['tumor_ratio'].values[cluster_indices]
        avg_tumor = np.nanmean(tumor_ratios)
        
        # Calculate the average connective ratio
        conn_ratios = connective_ratios[cluster_indices]
        avg_connective = np.nanmean(conn_ratios)
        
        # Overall score
        score = -avg_tumor + avg_connective
        
        cluster_stats[cluster_id] = {
            'avg_tumor_ratio': avg_tumor,
            'avg_connective_ratio': avg_connective,
            'size': len(cluster_indices),
            'score': score,
            'indices': cluster_indices
        }
        
    if not cluster_stats:
        print(f"No valid clusters")
        return None
    
    # Select the cluster with the highest comprehensive score. If there are multiple clusters with the same highest score, choose the one with the largest number of spots.
    max_score = max(stats['score'] for stats in cluster_stats.values())
    candidates = []
    for cluster_id, stats in cluster_stats.items():
        if abs(stats['score'] - max_score) < 1e-6:
            candidates.append((cluster_id, stats))
    
    if len(candidates) == 1:
        best_cluster_id = candidates[0][0]
        best_stats = candidates[0][1]
    else:
        best = max(candidates, key=lambda x: x[1]['size'])
        best_cluster_id = best[0]
        best_stats = best[1]
    
    # In the selected cluster, the root node is also selected using the comprehensive score.
    cluster_indices = best_stats['indices']
    cluster_tumor_ratios = adata_gene.obs['tumor_ratio'].values[cluster_indices]
    cluster_conn_ratios = connective_ratios[cluster_indices]

    spot_scores = -cluster_tumor_ratios + cluster_conn_ratios
    
    max_spot_score = np.nanmax(spot_scores)
    if np.isnan(max_spot_score):
        min_tumor = np.nanmin(cluster_tumor_ratios)
        min_tumor_mask = cluster_tumor_ratios == min_tumor
        candidate_indices = cluster_indices[min_tumor_mask]
    else:
        candidate_mask = spot_scores == max_spot_score
        candidate_indices = cluster_indices[candidate_mask]
    
    # If there are multiple candidates, select the one with the smallest value of Diffmap1.
    if len(candidate_indices) > 1:
        if diffmap and 'X_diffmap' in adata_gene.obsm and adata_gene.obsm['X_diffmap'].shape[1] > 0:
            comp_arr = adata_gene.obsm['X_diffmap'][:, 0]
        else:
            comp_arr = adata_gene.obsm['X_pca'][:, 0]
        chosen_root_idx = candidate_indices[np.argmin(comp_arr[candidate_indices])]

    else:
        chosen_root_idx = candidate_indices[0] if len(candidate_indices) > 0 else None
    
    if chosen_root_idx is None:
        return None
    
    root_idx = int(chosen_root_idx)
    
    print(f"Root node index: {root_idx}, barcode: {adata_gene.obs_names[root_idx]}")    
    return root_idx

# ====================Select the root node based on the image characteristics ==================== 
def select_root_from_image_features(adata_img, adata_gene, cell_composition_df, sample_name, resolution=RESOLUTION, diffmap=DIFFMAP):

    if 'X_pca' not in adata_img.obsm:
        sc.tl.pca(adata_img, n_comps=N_PCS, random_state=0)
    
    if 'connectivities' not in adata_img.obsp:
        sc.pp.neighbors(adata_img, n_pcs=N_PCS, random_state=0)
    
    if diffmap and 'X_diffmap' not in adata_img.obsm:
        sc.tl.diffmap(adata_img)
        sc.pp.neighbors(adata_img, use_rep='X_diffmap', random_state=0)
    
    if 'leiden' not in adata_img.obs.columns:
        sc.tl.leiden(adata_img, resolution=resolution, random_state=0)
    
    leiden_clusters = adata_img.obs['leiden'].unique()
    spot_barcodes = adata_gene.obs_names.astype(str).tolist()
    barcode_to_idx = {bc: i for i, bc in enumerate(spot_barcodes)}
    
    # Calculate the proportion of connectives
    connective_ratios = compute_connective_ratios(adata_gene, cell_composition_df)
    if connective_ratios is None:
        return None
    
    # Calculate the comprehensive indicators for each cluster
    cluster_stats = {}
    for cluster_id in leiden_clusters:
        cluster_mask = adata_img.obs['leiden'] == cluster_id
        cluster_img_indices = np.where(cluster_mask)[0]
        
        cluster_barcodes = adata_img.obs_names[cluster_mask].astype(str).tolist()
        cluster_gene_indices = []
        for bc in cluster_barcodes:
            if bc in barcode_to_idx:
                cluster_gene_indices.append(barcode_to_idx[bc])
        
        if len(cluster_gene_indices) == 0:
            continue  

        cluster_gene_indices = np.array(cluster_gene_indices)
        tumor_ratios = adata_gene.obs['tumor_ratio'].values[cluster_gene_indices]
        avg_tumor = np.nanmean(tumor_ratios)
        conn_ratios = connective_ratios[cluster_gene_indices]
        avg_connective = np.nanmean(conn_ratios)
        score = -avg_tumor + avg_connective
        
        cluster_stats[cluster_id] = {
            'avg_tumor_ratio': avg_tumor,
            'avg_connective_ratio': avg_connective,
            'size': len(cluster_gene_indices),
            'score': score,
            'gene_indices': cluster_gene_indices,
            'img_indices': np.array(cluster_img_indices)
        }

        return None
    
    # Select the cluster with the highest comprehensive score (in case of a tie, choose the one with the most spots)
    max_score = max(stats['score'] for stats in cluster_stats.values())
    candidates = []
    for cluster_id, stats in cluster_stats.items():
        if abs(stats['score'] - max_score) < 1e-6:
            candidates.append((cluster_id, stats))
    
    if len(candidates) == 1:
        best_cluster_id = candidates[0][0]
        best_stats = candidates[0][1]
    else:
        best = max(candidates, key=lambda x: x[1]['size'])
        best_cluster_id = best[0]
        best_stats = best[1]
   
    # In the selected cluster, the root node is selected based on the comprehensive score.
    cluster_gene_indices = best_stats['gene_indices']  
    
    if not isinstance(cluster_gene_indices, np.ndarray):
        cluster_gene_indices = np.array(cluster_gene_indices)
    
    cluster_tumor_ratios = adata_gene.obs['tumor_ratio'].values[cluster_gene_indices]
    cluster_conn_ratios = connective_ratios[cluster_gene_indices]
    
    spot_scores = -cluster_tumor_ratios + cluster_conn_ratios
    
    max_spot_score = np.nanmax(spot_scores)
    if np.isnan(max_spot_score):
        min_tumor = np.nanmin(cluster_tumor_ratios)
        min_tumor_mask = cluster_tumor_ratios == min_tumor
        candidate_indices = cluster_gene_indices[min_tumor_mask]
    else:
        candidate_mask = spot_scores == max_spot_score
        candidate_indices = cluster_gene_indices[candidate_mask]
   
    # If there are multiple candidates, select the one with the smallest value of Diffmap1.
    if len(candidate_indices) > 1:
        if diffmap and 'X_diffmap' in adata_gene.obsm and adata_gene.obsm['X_diffmap'].shape[1] > 0:
            comp_arr = adata_gene.obsm['X_diffmap'][:, 0]
        else:
            comp_arr = adata_gene.obsm['X_pca'][:, 0]
        root_idx = candidate_indices[np.argmin(comp_arr[candidate_indices])]
    else:
        root_idx = candidate_indices[0] if len(candidate_indices) > 0 else None

    if root_idx is None:
        return None
    
    root_idx = int(root_idx)
    print(f"Root node index:{root_idx}, barcode: {adata_gene.obs_names[root_idx]}")
    return root_idx

# ==================== Correlation statistical function ==================== 
def compute_pc_celltype_correlations(pca_result, cell_type_ratio_pivot, valid_cell_types,sample_name, save_dir, modality="gene", gene_set="",pca_variance_ratio=None):
    n_pcs = min(4, pca_result.shape[1])
    pc_names = [f'PC{i+1}' for i in range(n_pcs)]
    results = []
    for cell_type in valid_cell_types:
        ratios = cell_type_ratio_pivot[cell_type].values
        
        for pc_idx, pc_name in enumerate(pc_names):
            pc_values = pca_result[:, pc_idx]
            
            valid_mask = np.isfinite(ratios) & np.isfinite(pc_values)
            if valid_mask.sum() > 3:
                corr, p_val = spearmanr(pc_values[valid_mask], ratios[valid_mask])
                results.append({
                    'cell_type': cell_type,
                    'PC': pc_name,
                    'spearman_r': corr,
                    'pvalue': p_val,
                    'n_spots': valid_mask.sum()
                })
    results_df = pd.DataFrame(results)
    if pca_variance_ratio is not None:
        variance_df = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(min(4, len(pca_variance_ratio)))],
            'explained_variance_ratio': pca_variance_ratio[:min(4, len(pca_variance_ratio))],
            'explained_variance_percent': pca_variance_ratio[:min(4, len(pca_variance_ratio))] * 100
        })
    else:
        # Calculate the proportion of variance explained from pca_result
        from sklearn.decomposition import PCA
        pca_temp = PCA(n_components=min(4, pca_result.shape[1]))
        pca_temp.fit(pca_result)
        var_ratio = pca_temp.explained_variance_ratio_
        variance_df = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(len(var_ratio))],
            'explained_variance_ratio': var_ratio,
            'explained_variance_percent': var_ratio * 100
        })
    
    suffix = f"_{gene_set}" if gene_set else ""
    var_path = os.path.join(save_dir, f"{sample_name}_{modality}{suffix}_pc_variance.csv")
    variance_df.to_csv(var_path, index=False, encoding='utf-8-sig')
    corr_path = os.path.join(save_dir, f"{sample_name}_{modality}{suffix}_pc_celltype_correlations.csv")
    results_df.to_csv(corr_path, index=False, encoding='utf-8-sig')
    return results_df, variance_df

def compute_celltype_pseudotime_correlations(pseudotime, cell_type_ratio_pivot, valid_cell_types,sample_name, save_dir, modality="gene"):
    results = []
    for cell_type in valid_cell_types:
        ratios = cell_type_ratio_pivot[cell_type].values
        
        valid_mask = np.isfinite(pseudotime) & np.isfinite(ratios)
        if valid_mask.sum() > 3:
            corr, p_val = spearmanr(pseudotime[valid_mask], ratios[valid_mask])
            results.append({
                'cell_type': cell_type,
                'spearman_r': corr,
                'pvalue': p_val,
                'n_spots': valid_mask.sum()
            })
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('spearman_r', ascending=False)
    corr_path = os.path.join(save_dir, f"{sample_name}_{modality}_celltype_pseudotime_correlations.csv")
    results_df.to_csv(corr_path, index=False, encoding='utf-8-sig')
    return results_df

def save_all_correlation_summary(corr_hvg, corr_svg, corr_img,corr_ptime_gene, corr_ptime_img,var_hvg, var_svg, var_img,sample_name, save_dir):

    if var_hvg is not None:
        var_hvg.to_csv(os.path.join(save_dir, f"{sample_name}_HVG_PC_Variance.csv"), index=False)
    if var_svg is not None:
        var_svg.to_csv(os.path.join(save_dir, f"{sample_name}_SVG_PC_Variance.csv"), index=False) 
    if var_img is not None:
        var_img.to_csv(os.path.join(save_dir, f"{sample_name}_Image_PC_Variance.csv"), index=False)
    if corr_hvg is not None and not corr_hvg.empty:
        pivot_hvg = corr_hvg.pivot(index='cell_type', columns='PC', values='spearman_r')
        pivot_hvg.to_csv(os.path.join(save_dir, f"{sample_name}_HVG_PC_CellType.csv"))
    if corr_svg is not None and not corr_svg.empty:
        pivot_svg = corr_svg.pivot(index='cell_type', columns='PC', values='spearman_r')
        pivot_svg.to_csv(os.path.join(save_dir, f"{sample_name}_SVG_PC_CellType.csv"))
    if corr_img is not None and not corr_img.empty:
        pivot_img = corr_img.pivot(index='cell_type', columns='PC', values='spearman_r')
        pivot_img.to_csv(os.path.join(save_dir, f"{sample_name}_Image_PC_CellType.csv"))
    if corr_ptime_gene is not None and not corr_ptime_gene.empty:
        corr_ptime_gene.to_csv(os.path.join(save_dir, f"{sample_name}_CellType_GenePtime.csv"), index=False)
    if corr_ptime_img is not None and not corr_ptime_img.empty:
        corr_ptime_img.to_csv(os.path.join(save_dir, f"{sample_name}_CellType_ImagePtime.csv"), index=False)

# ==================== Calculate and save the correlation between cell proportions and pseudo-time ==================== 
def compute_celltype_pseudotime_correlations_full(adata_gene, adata_img, cell_composition_df,gene_ptime, img_ptime, sample_name, save_dir):

    if cell_composition_df is None or cell_composition_df.empty:
        return
    spot_barcodes = adata_gene.obs_names.astype(str).tolist()
    spot_total = cell_composition_df.groupby('spot_barcode')['cell_id'].nunique().reset_index()
    spot_total.columns = ['spot_barcode', 'total_cells']
    all_cell_types = cell_composition_df['cell_type'].unique()
    cell_type_ratios = {}
    for cell_type in all_cell_types:
        cell_type_filtered = cell_composition_df[
            cell_composition_df['cell_type'] == cell_type
        ]
        spot_counts = cell_type_filtered.groupby('spot_barcode')['cell_id'].nunique().reset_index()
        spot_counts.columns = ['spot_barcode', f'{cell_type}_count']
        
        spot_merged = spot_total.merge(spot_counts, on='spot_barcode', how='left')
        spot_merged[f'{cell_type}_count'] = spot_merged[f'{cell_type}_count'].fillna(0)
        spot_merged[f'{cell_type}_ratio'] = spot_merged[f'{cell_type}_count'] / spot_merged['total_cells']
        
        barcode_to_ratio = dict(zip(spot_merged['spot_barcode'], spot_merged[f'{cell_type}_ratio']))
        cell_type_ratios[cell_type] = np.array([barcode_to_ratio.get(bc, 0.0) for bc in spot_barcodes])

    celltype_df = pd.DataFrame({
        'spot_barcode': spot_barcodes,
        'spatial_x': adata_gene.obsm['spatial'][:, 0] if 'spatial' in adata_gene.obsm else np.nan,
        'spatial_y': adata_gene.obsm['spatial'][:, 1] if 'spatial' in adata_gene.obsm else np.nan,
        'tumor_ratio': adata_gene.obs['tumor_ratio'].values,
        'total_cells': adata_gene.obs['cell_count'].values,
        'gene_pseudotime': gene_ptime,
        'image_pseudotime': img_ptime if img_ptime is not None else np.nan,
        'leiden_cluster': adata_gene.obs['leiden'].values if 'leiden' in adata_gene.obs.columns else np.nan
    })
    
    for cell_type, ratios in cell_type_ratios.items():
        celltype_df[f'{cell_type}_ratio'] = ratios

    celltype_path = os.path.join(save_dir, f"{sample_name}_spot_celltype_ratios.csv")
    celltype_df.to_csv(celltype_path, index=False, encoding='utf-8-sig')
    results = []

    gene_ptime_valid = gene_ptime
    img_ptime_valid = img_ptime if img_ptime is not None else np.full_like(gene_ptime, np.nan)
    for cell_type in all_cell_types:
        ratios = cell_type_ratios[cell_type]
        valid_mask = np.isfinite(ratios) & np.isfinite(gene_ptime_valid)
        if valid_mask.sum() > 3:
            pearson_r, pearson_p = pearsonr(ratios[valid_mask], gene_ptime_valid[valid_mask])
            spearman_r, spearman_p = spearmanr(ratios[valid_mask], gene_ptime_valid[valid_mask])
            results.append({
                'cell_type': cell_type,
                'pseudotime_type': 'gene',
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_spots': valid_mask.sum()
            })
        else:
            results.append({
                'cell_type': cell_type,
                'pseudotime_type': 'gene',
                'pearson_r': np.nan,
                'pearson_p': np.nan,
                'spearman_r': np.nan,
                'spearman_p': np.nan,
                'n_spots': valid_mask.sum()
            })
        
        if img_ptime is not None:
            valid_mask_img = np.isfinite(ratios) & np.isfinite(img_ptime_valid)
            if valid_mask_img.sum() > 3:
                pearson_r, pearson_p = pearsonr(ratios[valid_mask_img], img_ptime_valid[valid_mask_img])
                spearman_r, spearman_p = spearmanr(ratios[valid_mask_img], img_ptime_valid[valid_mask_img])
                results.append({
                    'cell_type': cell_type,
                    'pseudotime_type': 'image',
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'n_spots': valid_mask_img.sum()
                })
            else:
                results.append({
                    'cell_type': cell_type,
                    'pseudotime_type': 'image',
                    'pearson_r': np.nan,
                    'pearson_p': np.nan,
                    'spearman_r': np.nan,
                    'spearman_p': np.nan,
                    'n_spots': valid_mask_img.sum()
                })
    
    corr_df = pd.DataFrame(results)
    corr_path = os.path.join(save_dir, f"{sample_name}_celltype_pseudotime_correlations.csv")
    corr_df.to_csv(corr_path, index=False, encoding='utf-8-sig')
    return celltype_df, corr_df

# ==================== Visualization of PCA result graph ====================
def plot_pca_pseudotime_with_cell_types(adata, pca_result, pseudotime, cell_composition_df,sample_name, save_dir, modality="image", gene_set="",figsize=None, s=20, root_idx=None):
    if modality == "gene":
        modality_label = "Gene Expression"
        ptime_label = "Gene Pseudotime"
    else:
        modality_label = "Image Feature"
        ptime_label = "Image Pseudotime"

    if gene_set:
        modality_label = f"{modality_label} ({gene_set})"

    pc1 = pca_result[:, 0]
    pc2 = pca_result[:, 1]
    pc3 = pca_result[:, 2] if pca_result.shape[1] > 2 else np.zeros_like(pc1)
    pc4 = pca_result[:, 3] if pca_result.shape[1] > 3 else np.zeros_like(pc1)

    if (
        hasattr(adata, "uns")
        and "pca" in adata.uns
        and "variance_ratio" in adata.uns["pca"]
    ):
        var_ratio = adata.uns["pca"]["variance_ratio"]
        pc1_var = var_ratio[0] * 100 if len(var_ratio) > 0 else 0
        pc2_var = var_ratio[1] * 100 if len(var_ratio) > 1 else 0
        pc3_var = var_ratio[2] * 100 if len(var_ratio) > 2 else 0
        pc4_var = var_ratio[3] * 100 if len(var_ratio) > 3 else 0
    else:
        pc1_var, pc2_var, pc3_var, pc4_var = 0, 0, 0, 0

    pc_pairs = [
        (pc1, pc2, f"PC1 ({pc1_var:.1f}%)", f"PC2 ({pc2_var:.1f}%)", "PC1 vs PC2"),
        (pc1, pc3, f"PC1 ({pc1_var:.1f}%)", f"PC3 ({pc3_var:.1f}%)", "PC1 vs PC3"),
        (pc1, pc4, f"PC1 ({pc1_var:.1f}%)", f"PC4 ({pc4_var:.1f}%)", "PC1 vs PC4"),
    ]

    # Calculate cell-type proportions for each spot
    if cell_composition_df is not None and not cell_composition_df.empty:
        spot_cell_counts = (
            cell_composition_df
            .groupby(["spot_barcode", "cell_type"])
            .size()
            .reset_index(name="count")
        )

        spot_total_counts = (
            spot_cell_counts
            .groupby("spot_barcode")["count"]
            .sum()
            .reset_index(name="total_count")
        )

        spot_cell_ratios = spot_cell_counts.merge(
            spot_total_counts, on="spot_barcode"
        )
        spot_cell_ratios["ratio"] = (
            spot_cell_ratios["count"] / spot_cell_ratios["total_count"]
        )

        cell_type_ratio_pivot = (
            spot_cell_ratios
            .pivot_table(
                index="spot_barcode",
                columns="cell_type",
                values="ratio",
                fill_value=0,
            )
            .reset_index()
        )

        barcodes = adata.obs_names.astype(str).tolist()
        cell_type_ratio_pivot = (
            cell_type_ratio_pivot
            .set_index("spot_barcode")
            .reindex(barcodes)
            .fillna(0)
            .reset_index()
        )

        all_cell_types = [
            col for col in cell_type_ratio_pivot.columns
            if col != "spot_barcode" and col.lower() != "unknown"
        ]

        valid_cell_types = [
            ct for ct in all_cell_types
            if cell_type_ratio_pivot[ct].sum() > 0
        ]
    else:
        valid_cell_types = []

    n_rows = 1 + len(valid_cell_types)
    n_cols = 3

    if figsize is None:
        figsize = (18, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Row 1: PCA colored by pseudotime
    for col_idx, (x, y, xlabel, ylabel, title) in enumerate(pc_pairs):
        ax = axes[0, col_idx]

        valid_mask = np.isfinite(pseudotime) & np.isfinite(x) & np.isfinite(y)

        if valid_mask.sum() > 3:
            corr_x, _ = spearmanr(x[valid_mask], pseudotime[valid_mask])
            corr_y, _ = spearmanr(y[valid_mask], pseudotime[valid_mask])

            title = (
                f"{title}\n"
                f"Corr with {ptime_label}: "
                f"PC1={corr_x:.3f}, PC{col_idx + 2}={corr_y:.3f}"
            )

        scatter = ax.scatter(
            x, y,
            c=pseudotime,
            cmap="viridis",
            s=s,
            alpha=0.7,
            edgecolors="none",
        )

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label(ptime_label, fontsize=9)

    # Remaining rows: PCA colored by cell-type proportions
    for row_idx, cell_type in enumerate(valid_cell_types, start=1):
        ratios = cell_type_ratio_pivot[cell_type].values

        for col_idx, (x, y, xlabel, ylabel, title) in enumerate(pc_pairs):
            ax = axes[row_idx, col_idx]

            scatter = ax.scatter(
                x, y,
                c=ratios,
                cmap="Reds",
                s=s,
                alpha=0.7,
                edgecolors="none",
                vmin=0,
                vmax=1,
            )

            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)

            display_name = cell_type.replace("_", " ").title()
            if len(display_name) > 20:
                display_name = display_name[:20] + "..."

            ax.set_title(
                f"{display_name} Ratio",
                fontsize=11,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)

            cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
            cbar.set_label("Ratio", fontsize=9)

    fig.suptitle(
        f"{sample_name} - {modality_label} PCA\n"
        f"Row 1: {ptime_label} | "
        f"Rows 2-{n_rows}: Cell Type Ratio Coloring",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)

    suffix = f"_{gene_set}" if gene_set else ""
    save_path = os.path.join(
        save_dir,
        f"{sample_name}_{modality}{suffix}_pca_pseudotime_cell_types.png",
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    return fig

def plot_celltype_pca_combined_hvg(adata, pca_result, pseudotime, cell_composition_df,sample_name, save_dir, modality="gene", gene_set="HVG",figsize=(18, 6), s=20, root_idx=None):

    modality_label = "Gene Expression"
    ptime_label = "Gene Pseudotime"

    if gene_set:
        modality_label = f"{modality_label} ({gene_set})"
    pc1 = pca_result[:, 0]
    pc2 = pca_result[:, 1]

    if hasattr(adata, 'uns') and 'pca' in adata.uns and 'variance_ratio' in adata.uns['pca']:
        var_ratio = adata.uns['pca']['variance_ratio']
        pc1_var = var_ratio[0] * 100 if len(var_ratio) > 0 else 0
        pc2_var = var_ratio[1] * 100 if len(var_ratio) > 1 else 0
    else:
        pc1_var, pc2_var = 0, 0

    if cell_composition_df is not None and not cell_composition_df.empty:
        spot_cell_counts = cell_composition_df.groupby(['spot_barcode', 'cell_type']).size().reset_index(name='count')
        spot_total_counts = spot_cell_counts.groupby('spot_barcode')['count'].sum().reset_index(name='total_count')
        spot_cell_ratios = spot_cell_counts.merge(spot_total_counts, on='spot_barcode')
        spot_cell_ratios['ratio'] = spot_cell_ratios['count'] / spot_cell_ratios['total_count']
        
        cell_type_ratio_pivot = spot_cell_ratios.pivot_table(
            index='spot_barcode', 
            columns='cell_type', 
            values='ratio',
            fill_value=0
        ).reset_index()
        
        barcodes = adata.obs_names.astype(str).tolist()
        cell_type_ratio_pivot = cell_type_ratio_pivot.set_index('spot_barcode').reindex(barcodes).fillna(0).reset_index()
        
        all_cell_types = [col for col in cell_type_ratio_pivot.columns 
                          if col != 'spot_barcode' and col.lower() != 'unknown']
        valid_cell_types = [ct for ct in all_cell_types if cell_type_ratio_pivot[ct].sum() > 0]
        
        display_types = []
        for ct in ['connective', 'neoplastic']:
            matching = [t for t in valid_cell_types if ct in t.lower()]
            if matching:
                display_types.append(matching[0])
        
        if len(display_types) < 2:
            display_types = valid_cell_types[:2]
        
        n_display = len(display_types)
    else:
        n_display = 0

    n_cols = 1 + n_display
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    ax = axes[0]
    valid_mask = np.isfinite(pseudotime) & np.isfinite(pc1) & np.isfinite(pc2)
    
    scatter = ax.scatter(pc1[valid_mask], pc2[valid_mask], 
                         c=pseudotime[valid_mask], cmap='viridis', 
                         s=s, alpha=0.8, edgecolors='none')
    
    ax.set_xlabel(f'PC1 ({pc1_var:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pc2_var:.1f}%)', fontsize=11)
    ax.set_title(f'{ptime_label}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(ptime_label, fontsize=10)

    for idx, cell_type in enumerate(display_types):
        ax = axes[idx + 1]
        ratios = cell_type_ratio_pivot[cell_type].values
        
        valid_mask_ct = np.isfinite(ratios) & np.isfinite(pc1) & np.isfinite(pc2)
        
        scatter = ax.scatter(pc1[valid_mask_ct], pc2[valid_mask_ct], 
                             c=ratios[valid_mask_ct], cmap='Reds', 
                             s=s, alpha=0.8, edgecolors='none',
                             vmin=0, vmax=1)
        
        ax.set_xlabel(f'PC1 ({pc1_var:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC2 ({pc2_var:.1f}%)', fontsize=11)
        
        display_name = cell_type.replace('_', ' ').capitalize()
        ax.set_title(f'{display_name} Ratio', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Ratio', fontsize=10)

    plt.suptitle(f'{sample_name} - {modality_label} PCA', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    suffix = f"_{gene_set}" if gene_set else ""
    save_path = os.path.join(save_dir, f"{sample_name}_{modality}{suffix}_celltype_combined_pca.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_celltype_pca_combined(adata, pca_result, pseudotime, cell_composition_df,sample_name, save_dir, modality="image", gene_set="",figsize=(18, 6), s=20, root_idx=None):
    
    if modality == "image":
        modality_label = "Image Feature"
        ptime_label = "Image Pseudotime"
    else:
        modality_label = "Gene Expression"
        ptime_label = "Gene Pseudotime" 
    if gene_set:
        modality_label = f"{modality_label} ({gene_set})"

    pc1 = pca_result[:, 0]
    pc2 = pca_result[:, 1]
    
    if hasattr(adata, 'uns') and 'pca' in adata.uns and 'variance_ratio' in adata.uns['pca']:
        var_ratio = adata.uns['pca']['variance_ratio']
        pc1_var = var_ratio[0] * 100 if len(var_ratio) > 0 else 0
        pc2_var = var_ratio[1] * 100 if len(var_ratio) > 1 else 0
    else:
        pc1_var, pc2_var = 0, 0

    if cell_composition_df is not None and not cell_composition_df.empty:
        spot_cell_counts = cell_composition_df.groupby(['spot_barcode', 'cell_type']).size().reset_index(name='count')
        spot_total_counts = spot_cell_counts.groupby('spot_barcode')['count'].sum().reset_index(name='total_count')
        spot_cell_ratios = spot_cell_counts.merge(spot_total_counts, on='spot_barcode')
        spot_cell_ratios['ratio'] = spot_cell_ratios['count'] / spot_cell_ratios['total_count']
        
        cell_type_ratio_pivot = spot_cell_ratios.pivot_table(
            index='spot_barcode', 
            columns='cell_type', 
            values='ratio',
            fill_value=0
        ).reset_index()
        
        barcodes = adata.obs_names.astype(str).tolist()
        cell_type_ratio_pivot = cell_type_ratio_pivot.set_index('spot_barcode').reindex(barcodes).fillna(0).reset_index()
        
        all_cell_types = [col for col in cell_type_ratio_pivot.columns 
                          if col != 'spot_barcode' and col.lower() != 'unknown']
        valid_cell_types = [ct for ct in all_cell_types if cell_type_ratio_pivot[ct].sum() > 0]

        display_types = []
        for ct in ['connective', 'neoplastic']:
            matching = [t for t in valid_cell_types if ct in t.lower()]
            if matching:
                display_types.append(matching[0])

        if len(display_types) < 2:
            display_types = valid_cell_types[:2]
        
        n_display = len(display_types)
    else:
        n_display = 0

    n_cols = 1 + n_display
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    ax = axes[0]
    valid_mask = np.isfinite(pseudotime) & np.isfinite(pc1) & np.isfinite(pc2)
    
    scatter = ax.scatter(pc1[valid_mask], pc2[valid_mask], 
                         c=pseudotime[valid_mask], cmap='viridis', 
                         s=s, alpha=0.8, edgecolors='none')
    
    ax.set_xlabel(f'PC1 ({pc1_var:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pc2_var:.1f}%)', fontsize=11)
    ax.set_title(f'{ptime_label}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(ptime_label, fontsize=10)

    for idx, cell_type in enumerate(display_types):
        ax = axes[idx + 1]
        ratios = cell_type_ratio_pivot[cell_type].values
        
        valid_mask_ct = np.isfinite(ratios) & np.isfinite(pc1) & np.isfinite(pc2)
        
        scatter = ax.scatter(pc1[valid_mask_ct], pc2[valid_mask_ct], 
                             c=ratios[valid_mask_ct], cmap='Reds', 
                             s=s, alpha=0.8, edgecolors='none',
                             vmin=0, vmax=1)
        
        ax.set_xlabel(f'PC1 ({pc1_var:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC2 ({pc2_var:.1f}%)', fontsize=11)
        
        display_name = cell_type.replace('_', ' ').capitalize()
        ax.set_title(f'{display_name} Ratio', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Ratio', fontsize=10)

    plt.suptitle(f'{sample_name} - {modality_label} PCA', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    suffix = f"_{gene_set}" if gene_set else ""
    save_path = os.path.join(save_dir, f"{sample_name}_{modality}{suffix}_celltype_combined_pca.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_combined_pca_with_celltypes_separate(adata, pca_result, pseudotime, cell_composition_df,sample_name, save_dir, modality="image",figsize=(8, 7), s=20, cell_types=None):

    if cell_types is None:
        cell_types = ['connective', 'neoplastic']

    if modality == "gene":
        modality_label = "Gene Expression"
        ptime_label = "Gene Pseudotime"
    else:
        modality_label = "Image Feature"
        ptime_label = "Image Pseudotime"

    pc1 = pca_result[:, 0]
    pc2 = pca_result[:, 1]
    pc3 = pca_result[:, 2] if pca_result.shape[1] > 2 else np.zeros_like(pc1)
    pc4 = pca_result[:, 3] if pca_result.shape[1] > 3 else np.zeros_like(pc1)

    if hasattr(adata, 'uns') and 'pca' in adata.uns and 'variance_ratio' in adata.uns['pca']:
        var_ratio = adata.uns['pca']['variance_ratio']
        pc1_var = var_ratio[0] * 100 if len(var_ratio) > 0 else 0
        pc2_var = var_ratio[1] * 100 if len(var_ratio) > 1 else 0
        pc3_var = var_ratio[2] * 100 if len(var_ratio) > 2 else 0
        pc4_var = var_ratio[3] * 100 if len(var_ratio) > 3 else 0
    else:
        pc1_var, pc2_var, pc3_var, pc4_var = 0, 0, 0, 0

    pc_pairs = [
        (pc1, pc2, f'PC1 ({pc1_var:.1f}%)', f'PC2 ({pc2_var:.1f}%)', 'PC1_vs_PC2'),
        (pc1, pc3, f'PC1 ({pc1_var:.1f}%)', f'PC3 ({pc3_var:.1f}%)', 'PC1_vs_PC3'),
        (pc1, pc4, f'PC1 ({pc1_var:.1f}%)', f'PC4 ({pc4_var:.1f}%)', 'PC1_vs_PC4')
    ]

    spot_barcodes = adata.obs_names.astype(str).tolist()
    if cell_composition_df is not None and not cell_composition_df.empty:
        spot_total = cell_composition_df.groupby('spot_barcode')['cell_id'].nunique().reset_index()
        spot_total.columns = ['spot_barcode', 'total_cells']
        
        cell_type_ratios = {}
        for cell_type in cell_types:
            cell_type_filtered = cell_composition_df[
                cell_composition_df['cell_type'].str.contains(cell_type, case=False, na=False)
            ]
            spot_counts = cell_type_filtered.groupby('spot_barcode')['cell_id'].nunique().reset_index()
            spot_counts.columns = ['spot_barcode', f'{cell_type}_count']
            
            spot_merged = spot_total.merge(spot_counts, on='spot_barcode', how='left')
            spot_merged[f'{cell_type}_count'] = spot_merged[f'{cell_type}_count'].fillna(0)
            spot_merged[f'{cell_type}_ratio'] = spot_merged[f'{cell_type}_count'] / spot_merged['total_cells']
            
            barcode_to_ratio = dict(zip(spot_merged['spot_barcode'], spot_merged[f'{cell_type}_ratio']))
            cell_type_ratios[cell_type] = np.array([barcode_to_ratio.get(bc, 0.0) for bc in spot_barcodes])
        
        connective_ratios = cell_type_ratios.get('connective', np.zeros(len(spot_barcodes)))
        neoplastic_ratios = cell_type_ratios.get('neoplastic', np.zeros(len(spot_barcodes)))
        ratio_diff = neoplastic_ratios - connective_ratios

        all_pc_values = np.concatenate([pc1, pc2, pc3, pc4])
        pc_min, pc_max = np.percentile(all_pc_values, [1, 99])
        pc_range = pc_max - pc_min
        x_limits = (pc_min - pc_range * 0.1, pc_max + pc_range * 0.1)
    else:
        connective_ratios = np.zeros(len(spot_barcodes))
        neoplastic_ratios = np.zeros(len(spot_barcodes))
        ratio_diff = np.zeros(len(spot_barcodes))
        x_limits = (pc1.min() - 0.5, pc1.max() + 0.5)

    cmap = plt.cm.RdBu_r
    vmin, vmax = -1, 1

    for col_idx, (x, y, xlabel, ylabel, name) in enumerate(pc_pairs):
        fig, ax = plt.subplots(figsize=figsize)
        
        valid_mask = np.isfinite(pseudotime) & np.isfinite(x) & np.isfinite(y)
        scatter = ax.scatter(x[valid_mask], y[valid_mask], 
                            c=pseudotime[valid_mask], 
                            cmap='viridis', 
                            s=s, alpha=0.35, edgecolors='none')
        
        ax.set_xlim(x_limits)
        ax.set_ylim(x_limits)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=6)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(1.5)
        
        ax.set_xlabel('')
        ax.set_ylabel('')

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=16)
        cbar.outline.set_visible(False)
        cbar.solids.set_alpha(1.0)
        save_path = os.path.join(save_dir, f"{sample_name}_{modality}_{name}_pseudotime.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    for col_idx, (x, y, xlabel, ylabel, name) in enumerate(pc_pairs):
        fig, ax = plt.subplots(figsize=figsize)
        valid_mask = np.isfinite(ratio_diff) & np.isfinite(x) & np.isfinite(y)
        scatter = ax.scatter(x[valid_mask], y[valid_mask], 
                            c=ratio_diff[valid_mask], 
                            cmap=cmap, 
                            s=s, alpha=0.35,  
                            edgecolors='none',
                            vmin=vmin, vmax=vmax)
        
        ax.set_xlim(x_limits)
        ax.set_ylim(x_limits)
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(1.5)
        ax.set_xlabel('')
        ax.set_ylabel('')

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=16)
        cbar.outline.set_visible(False)
        cbar.solids.set_alpha(1.0)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.set_ticklabels(['Connective', '', 'Equal', '', 'Neoplastic'])
        save_path = os.path.join(save_dir, f"{sample_name}_{modality}_{name}_celltypes.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    return

# ==================== Auxiliary function ====================
def read_sample_list(file_path):
    """Read sample IDs from .txt (one ID per line) or .csv.
    CSV accepts a ``sample``/``sample_id``/``sample_name`` column, or uses
    the first column if none of those headers is present. A headerless,
    single-column CSV is also supported.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sample list not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        with path.open("r", encoding="utf-8-sig") as handle:
            samples = [line.strip() for line in handle
                       if line.strip() and not line.lstrip().startswith("#")]
    elif suffix == ".csv":
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", comment="#")
        if frame.empty and len(frame.columns) == 1:
            samples = [str(frame.columns[0]).strip()]
        else:
            matching = [column for column in frame.columns
                        if column.strip().lower() in {"sample", "sample_id", "sample_name"}]
            column = matching[0] if matching else frame.columns[0]
            samples = frame[column].dropna().astype(str).str.strip().tolist()
            # Headerless single-column CSV: pandas treats the first ID as the header.
            if not matching and len(frame.columns) == 1:
                samples.insert(0, str(frame.columns[0]).strip())
    else:
        raise ValueError("Sample list must be a .csv or .txt file")

    samples = list(dict.fromkeys(item for item in samples if item))
    if not samples:
        raise ValueError(f"Sample list is empty: {path}")
    return samples

def parse_arguments():
    """Configure experiment inputs without changing analysis code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-list", default="select_samples.csv", metavar="FILE",
                        help="CSV/TXT list of sample IDs.")
    parser.add_argument("--root-dir", default=ROOT_DIR, help="HEST dataset directory.")
    parser.add_argument("--save-root", default=SAVE_ROOT, help="Output directory.")
    parser.add_argument("--all-samples", action="store_true",
                        help="Use all sample directories instead of a sample list.")
    return parser.parse_args()

# ==================== Save the comprehensive results of the patch ====================
def save_patch_combined_results(adata_gene, adata_img, pca_result_gene, pca_result_img,gene_ptime, img_ptime, cell_composition_df,sample_name, save_dir, root_idx=None):
   
    n_spots = adata_gene.n_obs
    spot_barcodes = adata_gene.obs_names.astype(str).tolist()

    if 'spatial' in adata_gene.obsm:
        spatial_coords = adata_gene.obsm['spatial']
        x_coords = spatial_coords[:, 0]
        y_coords = spatial_coords[:, 1]
    else:
        x_coords, y_coords, _ = get_spatial_coords(adata_gene, sample_name)
    
    if pca_result_gene is not None:
        gene_pc1 = pca_result_gene[:, 0] if pca_result_gene.shape[1] > 0 else np.zeros(n_spots)
        gene_pc2 = pca_result_gene[:, 1] if pca_result_gene.shape[1] > 1 else np.zeros(n_spots)
        gene_pc3 = pca_result_gene[:, 2] if pca_result_gene.shape[1] > 2 else np.zeros(n_spots)
        gene_pc4 = pca_result_gene[:, 3] if pca_result_gene.shape[1] > 3 else np.zeros(n_spots)
    else:
        gene_pc1, gene_pc2, gene_pc3, gene_pc4 = np.zeros(n_spots), np.zeros(n_spots), np.zeros(n_spots), np.zeros(n_spots)

    if pca_result_img is not None:
        img_pc1 = pca_result_img[:, 0] if pca_result_img.shape[1] > 0 else np.zeros(n_spots)
        img_pc2 = pca_result_img[:, 1] if pca_result_img.shape[1] > 1 else np.zeros(n_spots)
        img_pc3 = pca_result_img[:, 2] if pca_result_img.shape[1] > 2 else np.zeros(n_spots)
        img_pc4 = pca_result_img[:, 3] if pca_result_img.shape[1] > 3 else np.zeros(n_spots)
    else:
        img_pc1, img_pc2, img_pc3, img_pc4 = np.zeros(n_spots), np.zeros(n_spots), np.zeros(n_spots), np.zeros(n_spots)

    tumor_ratios = adata_gene.obs['tumor_ratio'].values if 'tumor_ratio' in adata_gene.obs else np.zeros(n_spots)

    if cell_composition_df is not None and not cell_composition_df.empty:
        spot_total = cell_composition_df.groupby('spot_barcode')['cell_id'].nunique().reset_index()
        spot_total.columns = ['spot_barcode', 'total_cells']
        connective_filtered = cell_composition_df[
            cell_composition_df['cell_type'].str.contains('connective', case=False, na=False)
        ]
        spot_connective = connective_filtered.groupby('spot_barcode')['cell_id'].nunique().reset_index()
        spot_connective.columns = ['spot_barcode', 'connective_count']
        spot_ratio = spot_total.merge(spot_connective, on='spot_barcode', how='left')
        spot_ratio['connective_count'] = spot_ratio['connective_count'].fillna(0)
        spot_ratio['connective_ratio'] = spot_ratio['connective_count'] / spot_ratio['total_cells']
        barcode_to_connective = dict(zip(spot_ratio['spot_barcode'], spot_ratio['connective_ratio']))
        connective_ratios = np.array([barcode_to_connective.get(bc, 0.0) for bc in spot_barcodes])
    else:
        connective_ratios = np.zeros(n_spots)

    is_root = np.zeros(n_spots, dtype=bool)
    if root_idx is not None and 0 <= root_idx < n_spots:
        is_root[root_idx] = True
    
    leiden_cluster = adata_gene.obs['leiden'].values if 'leiden' in adata_gene.obs.columns else np.full(n_spots, np.nan)
    total_cells = adata_gene.obs['cell_count'].values if 'cell_count' in adata_gene.obs.columns else np.full(n_spots, np.nan)
 
    result_df = pd.DataFrame({
        'patch_id': range(1, n_spots + 1),  
        'barcode': spot_barcodes,
        'spatial_x': x_coords,
        'spatial_y': y_coords,
        'is_root': is_root, 
        'gene_PC1': gene_pc1,
        'gene_PC2': gene_pc2,
        'gene_PC3': gene_pc3,
        'gene_PC4': gene_pc4,
        'image_PC1': img_pc1,
        'image_PC2': img_pc2,
        'image_PC3': img_pc3,
        'image_PC4': img_pc4,
        'gene_pseudotime': gene_ptime,
        'image_pseudotime': img_ptime if img_ptime is not None else np.nan,
        'tumor_ratio': tumor_ratios,
        'connective_ratio': connective_ratios,
        'leiden_cluster': leiden_cluster,
        'total_cells': total_cells
    })
    
    save_path = os.path.join(save_dir, f"{sample_name}_patch_combined_results.csv")
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    variance_data = []
    
    if hasattr(adata_gene, 'uns') and 'pca' in adata_gene.uns and 'variance_ratio' in adata_gene.uns['pca']:
        var_ratio_gene = adata_gene.uns['pca']['variance_ratio']
        for i in range(min(4, len(var_ratio_gene))):
            variance_data.append({
                'PC': f'PC{i+1}',
                'gene_explained_variance_ratio': var_ratio_gene[i],
                'gene_explained_variance_percent': var_ratio_gene[i] * 100
            })
    else:
        for i in range(4):
            variance_data.append({
                'PC': f'PC{i+1}',
                'gene_explained_variance_ratio': np.nan,
                'gene_explained_variance_percent': np.nan
            })

    if hasattr(adata_img, 'uns') and 'pca' in adata_img.uns and 'variance_ratio' in adata_img.uns['pca']:
        var_ratio_img = adata_img.uns['pca']['variance_ratio']
        for i in range(min(4, len(var_ratio_img))):
            if i < len(variance_data):
                variance_data[i]['image_explained_variance_ratio'] = var_ratio_img[i]
                variance_data[i]['image_explained_variance_percent'] = var_ratio_img[i] * 100
    else:
        for i in range(4):
            if i < len(variance_data):
                variance_data[i]['image_explained_variance_ratio'] = np.nan
                variance_data[i]['image_explained_variance_percent'] = np.nan
    
    variance_df = pd.DataFrame(variance_data)
    var_path = os.path.join(save_dir, f"{sample_name}_pca_variance_ratios.csv")
    variance_df.to_csv(var_path, index=False, encoding='utf-8-sig')
    return result_df

# ==================== Save the UMAP dimensionality reduction results ====================
def save_umap_results(adata_gene, adata_img, gene_ptime, img_ptime, sample_name, save_dir):
    
    spot_barcodes = adata_gene.obs_names.astype(str).tolist()
    if 'spatial' in adata_gene.obsm:
        spatial_coords = adata_gene.obsm['spatial']
        x_coords = spatial_coords[:, 0]
        y_coords = spatial_coords[:, 1]
    else:
        x_coords, y_coords, _ = get_spatial_coords(adata_gene, sample_name)
    adata_gene_copy = adata_gene.copy()
    sc.pp.neighbors(adata_gene_copy, use_rep='X_pca')
    sc.tl.umap(adata_gene_copy, min_dist=0.5, spread=1.0)
    umap_gene = adata_gene_copy.obsm['X_umap']
    
    gene_umap_df = pd.DataFrame({
        'barcode': spot_barcodes,
        'spatial_x': x_coords,
        'spatial_y': y_coords,
        'UMAP1_gene': umap_gene[:, 0],
        'UMAP2_gene': umap_gene[:, 1],
        'gene_pseudotime': gene_ptime,
        'image_pseudotime': img_ptime if img_ptime is not None else np.nan,
        'tumor_ratio': adata_gene.obs['tumor_ratio'].values if 'tumor_ratio' in adata_gene.obs else np.nan,
        'leiden_cluster': adata_gene.obs['leiden'].values if 'leiden' in adata_gene.obs else np.nan
    })
    gene_umap_path = os.path.join(save_dir, f"{sample_name}_gene_umap_coordinates.csv")
    gene_umap_df.to_csv(gene_umap_path, index=False, encoding='utf-8-sig')
    adata_img_copy = adata_img.copy()
    sc.pp.neighbors(adata_img_copy, use_rep='X_pca')
    sc.tl.umap(adata_img_copy, min_dist=0.5, spread=1.0)
    umap_img = adata_img_copy.obsm['X_umap']
    
    img_umap_df = pd.DataFrame({
        'barcode': spot_barcodes,
        'spatial_x': x_coords,
        'spatial_y': y_coords,
        'UMAP1_image': umap_img[:, 0],
        'UMAP2_image': umap_img[:, 1],
        'gene_pseudotime': gene_ptime,
        'image_pseudotime': img_ptime if img_ptime is not None else np.nan,
        'tumor_ratio': adata_gene.obs['tumor_ratio'].values if 'tumor_ratio' in adata_gene.obs else np.nan,
        'leiden_cluster': adata_img.obs['leiden'].values if 'leiden' in adata_img.obs else np.nan
    })
    img_umap_path = os.path.join(save_dir, f"{sample_name}_image_umap_coordinates.csv")
    img_umap_df.to_csv(img_umap_path, index=False, encoding='utf-8-sig')
    combined_df = pd.DataFrame({
        'barcode': spot_barcodes,
        'spatial_x': x_coords,
        'spatial_y': y_coords,
        'UMAP1_gene': umap_gene[:, 0],
        'UMAP2_gene': umap_gene[:, 1],
        'UMAP1_image': umap_img[:, 0],
        'UMAP2_image': umap_img[:, 1],
        'gene_pseudotime': gene_ptime,
        'image_pseudotime': img_ptime if img_ptime is not None else np.nan,
        'tumor_ratio': adata_gene.obs['tumor_ratio'].values if 'tumor_ratio' in adata_gene.obs else np.nan,
        'leiden_cluster_gene': adata_gene.obs['leiden'].values if 'leiden' in adata_gene.obs else np.nan,
        'leiden_cluster_image': adata_img.obs['leiden'].values if 'leiden' in adata_img.obs else np.nan
    })
    combined_path = os.path.join(save_dir, f"{sample_name}_umap_coordinates_combined.csv")
    combined_df.to_csv(combined_path, index=False, encoding='utf-8-sig')
    return

# ============================ Main process ============================
if __name__ == "__main__":
    args = parse_arguments()
    ROOT_DIR = args.root_dir
    SAVE_ROOT = args.save_root
    os.makedirs(SAVE_ROOT, exist_ok=True)
    if not os.path.isdir(ROOT_DIR):
        raise NotADirectoryError(f"Dataset directory not found: {ROOT_DIR}")

    if args.all_samples:
        samples = sorted(d for d in os.listdir(ROOT_DIR)
                         if os.path.isdir(os.path.join(ROOT_DIR, d)))
    else:
        samples = read_sample_list(args.sample_list)

    print(f"The number of samples to be processed:{len(samples)}")
    print(f"Genetic selection model:{GENE_SELECTION_MODE}")
    print(f"DIFFMAP: {DIFFMAP}")
    print(f"Root node selection mode:{ROOT_SELECTION_MODE}")
    
    adata_img_list = []
    sample_names_list = []
    
    summary_columns = [
        "sample", "n_valid_spots", "n_spots_with_cells", "reason",
        "original_spots", "after_qc", "after_tissue_filter", "after_cell_filter", "after_all_filters",
        "retention_rate_qc", "retention_rate_tissue", "retention_rate_cell", "retention_rate_total",
        "final_cell_threshold", "leiden_clusters", "start_cluster_id", "start_cluster_tumor_ratio", "start_cluster_size",
        "gene_set_used", "n_genes_used",
        "pearson_r", "pearson_p", "kendall_tau", "kendall_p", "spearman_r", "spearman_p", "distance_corr",
        "kendall_tau_b", "kendall_tau_b_p", "tau_distance", "concordant_pairs", "discordant_pairs",
        "wasserstein_distance", "cosine_similarity",
        "tumor_ratio_mean", "tumor_ratio_max", "tumor_ratio_median",
        "cell_count_mean", "cell_count_max", "cell_count_median", "zero_cell_spots", "zero_cell_ratio",
        "total_cells_matched", "unique_cell_types", "avg_cells_per_spot",
        "gene_ptime_mean", "gene_ptime_std", "gene_ptime_min", "gene_ptime_max",
        "img_ptime_mean", "img_ptime_std", "img_ptime_min", "img_ptime_max"
    ]
    summary_rows = []
    
    for sample in tqdm(samples, desc="Samples"):
        sample_dir = os.path.join(ROOT_DIR, sample)
        st_path = os.path.join(sample_dir, "st", f"{sample}.h5ad")
        feat_path = os.path.join(sample_dir, "patches", f"{sample}_fix.h5")
        
        save_dir = os.path.join(SAVE_ROOT, sample)
        os.makedirs(save_dir, exist_ok=True)
        
        metrics = {col: np.nan for col in summary_columns if col not in ["sample", "reason"]}
        metrics["sample"] = sample
        metrics["reason"] = "OK"
        metrics["n_valid_spots"] = 0
        metrics["n_spots_with_cells"] = 0
        metrics["original_spots"] = 0
        metrics["after_qc"] = 0
        metrics["after_tissue_filter"] = 0
        metrics["after_cell_filter"] = 0
        metrics["after_all_filters"] = 0
        metrics["retention_rate_qc"] = 0.0
        metrics["retention_rate_tissue"] = 0.0
        metrics["retention_rate_cell"] = 0.0
        metrics["retention_rate_total"] = 0.0
        metrics["final_cell_threshold"] = 0
        metrics["leiden_clusters"] = 0
        metrics["start_cluster_id"] = ""
        metrics["start_cluster_tumor_ratio"] = np.nan
        metrics["start_cluster_size"] = 0
        metrics["zero_cell_spots"] = 0
        metrics["zero_cell_ratio"] = 0.0
        metrics["total_cells_matched"] = 0
        metrics["unique_cell_types"] = 0
        metrics["avg_cells_per_spot"] = 0.0
        metrics["gene_set_used"] = "Unknown"
        metrics["n_genes_used"] = 0

        if not os.path.exists(st_path):
            metrics["reason"] = "Missing ST file"
            print(f"Pass {sample}: {metrics['reason']}")
            summary_rows.append([metrics[col] for col in summary_columns])
            continue
        
        try:
            adata = sc.read_h5ad(st_path)
            metrics["original_spots"] = adata.n_obs
            print(f"\n--- Process the sample: {sample} ---")

            # ========== QC ==========
            sc.pp.filter_cells(adata, min_genes=100)
            sc.pp.filter_genes(adata, min_cells=3)
            adata.var["mt"] = adata.var_names.str.startswith("MT-")
            sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
            adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
            metrics["after_qc"] = adata.n_obs
            metrics["retention_rate_qc"] = metrics["after_qc"] / metrics["original_spots"] if metrics["original_spots"] > 0 else 0
            print(f"The number of spots after quality control:{adata.n_obs}")
            
            # ========== Organ contour filtering ==========
            poly, poly_path = load_polygon_for_sample(sample_dir, sample)
            if poly is not None:
                adata, inside_mask, before_tissue, after_tissue = filter_spots_by_tissue(adata, sample, sample_dir, save_dir)
                metrics["after_tissue_filter"] = after_tissue
                metrics["retention_rate_tissue"] = after_tissue / metrics["after_qc"] if metrics["after_qc"] > 0 else 0
            else:
                metrics["reason"] = "Missing tissue file"
                summary_rows.append([metrics[col] for col in summary_columns])
                continue
            
            # ========== Loading of cell division data ==========
            cell_polygons, cell_types, _ = load_cellvit_polygons(sample_dir, sample)
            seg_available = (cell_polygons is not None and len(cell_polygons) > 0)
            
            if not seg_available:
                metrics["reason"] = "Missing cell segmentation data"
                summary_rows.append([metrics[col] for col in summary_columns])
                continue
            
            # ========== Calculate the proportion of tumors ==========
            tumor_count, total_count, tumor_ratio, cell_composition_df = compute_spot_cell_counts_from_seg_fast_with_composition(
                adata, cell_polygons, cell_types, sample, sample_dir
            )
            
            adata.obs['tumor_ratio'] = tumor_ratio
            adata.obs['tumor_count'] = tumor_count
            adata.obs['cell_count'] = total_count
            
            before_cell_filter = adata.n_obs
            adata_before_cell_filter = adata.copy()
            
            adata, cell_count_mask, cell_counts, final_threshold = filter_spots_by_cell_count_enhanced(adata, sample)
            
            metrics["final_cell_threshold"] = final_threshold
            after_cell_filter = adata.n_obs
            metrics["after_cell_filter"] = after_cell_filter
            metrics["retention_rate_cell"] = after_cell_filter / before_cell_filter if before_cell_filter > 0 else 0
            metrics["n_spots_with_cells"] = adata.n_obs
            metrics["zero_cell_spots"] = np.sum(cell_counts == 0) if cell_counts is not None else 0
            metrics["zero_cell_ratio"] = metrics["zero_cell_spots"] / len(cell_counts) if cell_counts is not None else 0.0
            metrics["cell_count_mean"] = np.mean(cell_counts) if cell_counts is not None else np.nan
            metrics["cell_count_max"] = np.max(cell_counts) if cell_counts is not None else np.nan
            metrics["cell_count_median"] = np.median(cell_counts) if cell_counts is not None else np.nan
            
            if cell_composition_df is not None and not cell_composition_df.empty:
                filtered_spot_barcodes = adata.obs_names
                cell_composition_df_filtered = cell_composition_df[
                    cell_composition_df['spot_barcode'].isin(filtered_spot_barcodes)
                ]
                metrics["total_cells_matched"] = len(cell_composition_df_filtered)
                metrics["unique_cell_types"] = cell_composition_df_filtered['cell_type'].nunique()
                metrics["avg_cells_per_spot"] = cell_composition_df_filtered.groupby('spot_barcode')['cell_id'].nunique().mean()
                
                cell_composition_path = os.path.join(save_dir, f"{sample}_cell_composition.csv")
                cell_composition_df_filtered.to_csv(cell_composition_path, index=False, encoding="utf-8-sig")
            
            n_spots = adata.n_obs
            if n_spots < MIN_SPOTS:
                metrics["reason"] = f"Spot count out of range ({n_spots})"
                summary_rows.append([metrics[col] for col in summary_columns])
                continue
            
            # ========== Image feature matching ==========
            adata.layers['counts'] = adata.X.copy()
            adata = st.convert_scanpy(adata, use_quality='downscaled_fullres').copy()
            if 'spatial' not in adata.obsm:
                x_coords, y_coords, _ = get_spatial_coords(adata, sample)
                adata.obsm['spatial'] = np.vstack([x_coords, y_coords]).T
            
            if not os.path.exists(feat_path):
                metrics["reason"] = "Missing image feature file"
                summary_rows.append([metrics[col] for col in summary_columns])
                continue
            
            with h5py.File(feat_path, 'r') as f:
                feats = f[FEATURE_KEY][()]    
                barcodes_img = np.array(f[BARCODE_KEY][()]).squeeze().astype(str)
            feat_map = {bc: feats[i] for i, bc in enumerate(barcodes_img)}
            
            adata_bcs = adata.obs_names.astype(str)
            keep_idx = [i for i, bc in enumerate(adata_bcs) if bc in feat_map]
            if len(keep_idx) == 0:
                metrics["reason"] = "No matched barcodes"
                summary_rows.append([metrics[col] for col in summary_columns])
                continue
            
            adata_gene = adata[keep_idx, :].copy()
            img_feats = np.array([feat_map[bc] for bc in adata_gene.obs_names.astype(str)], dtype=np.float32)
            
            adata_img = sc.AnnData(img_feats)
            adata_img.obs = adata_gene.obs.copy()
            adata_img.obsm = adata_gene.obsm.copy()
            adata_img.uns = adata_gene.uns.copy()
            sc.pp.scale(adata_img)
            
            verify_spot_alignment(adata_gene, adata_img, sample, save_dir)
            
            metrics["after_all_filters"] = adata_gene.n_obs
            metrics["retention_rate_total"] = adata_gene.n_obs / metrics["original_spots"] if metrics["original_spots"] > 0 else 0
            
            # ========== Step 1: Preprocessing of gene expression data ==========
            sc.pp.normalize_total(adata_gene)
            sc.pp.log1p(adata_gene)
            adata_gene_log1p = adata_gene.copy()
            
            gene_set_used = "Unknown"
            n_genes_used = 0

            # ========== Step 2: Calculate the SVG gene ==========
            svg_genes, svg_results = get_top_svg_genes_for_sample(
                adata_gene_log1p, sample, n_neighbors=SVG_NEIGHBORS, top_n=SVG_TOP_N
            ) 
            svg_path = os.path.join(save_dir, f"{sample}_svg_genes.csv")
            svg_results.to_csv(svg_path, encoding='utf-8-sig')

            # ========== Step 3: Select the gene set based on the pattern ==========
            if GENE_SELECTION_MODE == "SVG":
                print(f"\n  [{sample}]Using SVG genetic data")

                if len(svg_genes) >= N_PCS:
                    adata_gene = adata_gene[:, svg_genes].copy()
                    gene_set_used = "SVG"
                    n_genes_used = adata_gene.n_vars
                else:       
                    sc.pp.highly_variable_genes(adata_gene, n_top_genes=HVG_N)
                    n_hvg = int(adata_gene.var['highly_variable'].sum())
                    if n_hvg < N_PCS:
                        metrics["reason"] = f"HVG count too low ({n_hvg} < {N_PCS})"
                        summary_rows.append([metrics[col] for col in summary_columns])
                        continue
                    adata_gene = adata_gene[:, adata_gene.var['highly_variable']].copy()
                    gene_set_used = "HVG (SVG fallback)"
                    n_genes_used = adata_gene.n_vars
            
            else:  # GENE_SELECTION_MODE == "HVG"
                print(f"\n  [{sample}] Using HVG gene data")
                sc.pp.highly_variable_genes(adata_gene, n_top_genes=HVG_N)
                n_hvg = int(adata_gene.var['highly_variable'].sum())
                if n_hvg < N_PCS:
                    metrics["reason"] = f"HVG count too low ({n_hvg} < {N_PCS})"
                    summary_rows.append([metrics[col] for col in summary_columns])
                    continue
                adata_gene = adata_gene[:, adata_gene.var['highly_variable']].copy()
                gene_set_used = "HVG"
                n_genes_used = adata_gene.n_vars
            sc.pp.scale(adata_gene)
            metrics["gene_set_used"] = gene_set_used
            metrics["n_genes_used"] = n_genes_used

            # ========== Step 4: Genetic Pattern - PCA + Diffmap + Leiden ==========
            sc.tl.pca(adata_gene, n_comps=N_PCS, random_state=0)
            sc.pp.neighbors(adata_gene, n_pcs=N_PCS, random_state=0)
            if DIFFMAP:
                sc.tl.diffmap(adata_gene)
                sc.pp.neighbors(adata_gene, use_rep='X_diffmap', random_state=0)
                print(f"use Diffmap neighbors")
            else:
                print(f"use PCA neighbors")
            sc.tl.leiden(adata_gene, resolution=RESOLUTION, random_state=0)
          
            # ========== Step 5: Image Mode - PCA + Diffmap + Leiden ==========
            sc.tl.pca(adata_img, n_comps=N_PCS, random_state=0)
            sc.pp.neighbors(adata_img, n_pcs=N_PCS, random_state=0)
            if DIFFMAP:
                sc.tl.diffmap(adata_img)
                sc.pp.neighbors(adata_img, use_rep='X_diffmap', random_state=0)
                print(f"use Diffmap neighbors")
            else:
                print(f"use PCA neighbors")         
            sc.tl.leiden(adata_img, resolution=RESOLUTION, random_state=0)

            # ========== Step 6: Selection of the root node ==========
            if seg_available:
                if ROOT_SELECTION_MODE == "image":
                    root_idx = select_root_from_image_features(
                        adata_img=adata_img,
                        adata_gene=adata_gene,
                        cell_composition_df=cell_composition_df,
                        sample_name=sample,
                        resolution=RESOLUTION,
                        diffmap=DIFFMAP
                    )
                    
                    if root_idx is None:
                        root_idx = select_root_from_gene_features(
                            adata_gene=adata_gene,
                            cell_composition_df=cell_composition_df,
                            sample_name=sample,
                            resolution=RESOLUTION,
                            diffmap=DIFFMAP
                        )
                        if root_idx is None:
                            metrics["reason"] = "Root selection failed"
                            summary_rows.append([metrics[col] for col in summary_columns])
                            continue
                        metrics["start_cluster_id"] = f"gene_fallback_{metrics['start_cluster_id']}"
                    else:
                        root_barcode = adata_gene.obs_names[root_idx]
                        img_idx = np.where(adata_img.obs_names == root_barcode)[0][0]
                        cluster_id = adata_img.obs['leiden'].iloc[img_idx]
                        cluster_mask = adata_img.obs['leiden'] == cluster_id
                        cluster_indices = np.where(cluster_mask)[0]
                        
                        cluster_barcodes = adata_img.obs_names[cluster_mask].astype(str).tolist()
                        cluster_tumor_ratios = []
                        for bc in cluster_barcodes:
                            if bc in adata_gene.obs_names:
                                idx = np.where(adata_gene.obs_names == bc)[0][0]
                                cluster_tumor_ratios.append(adata_gene.obs['tumor_ratio'].iloc[idx])
                        
                        metrics["start_cluster_id"] = f"image_{cluster_id}"
                        metrics["start_cluster_tumor_ratio"] = np.nanmean(cluster_tumor_ratios) if cluster_tumor_ratios else np.nan
                        metrics["start_cluster_size"] = len(cluster_indices)
                
                else:  # ROOT_SELECTION_MODE == "gene"
                    root_idx = select_root_from_gene_features(
                        adata_gene=adata_gene,
                        cell_composition_df=cell_composition_df,
                        sample_name=sample,
                        resolution=RESOLUTION,
                        diffmap=DIFFMAP
                    )
                    
                    if root_idx is None:
                        metrics["reason"] = "Root selection failed"
                        summary_rows.append([metrics[col] for col in summary_columns])
                        continue
                    
                    root_barcode = adata_gene.obs_names[root_idx]
                    cluster_id = adata_gene.obs['leiden'].iloc[root_idx]
                    cluster_mask = adata_gene.obs['leiden'] == cluster_id
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    cluster_tumor_ratios = adata_gene.obs['tumor_ratio'].values[cluster_indices]
                    metrics["start_cluster_id"] = f"gene_{cluster_id}"
                    metrics["start_cluster_tumor_ratio"] = np.nanmean(cluster_tumor_ratios) if len(cluster_tumor_ratios) > 0 else np.nan
                    metrics["start_cluster_size"] = len(cluster_indices)
            else:
                metrics["reason"] = "Missing cell_seg data file"
                summary_rows.append([metrics[col] for col in summary_columns])
                continue
          
            root_idx = int(root_idx)
            x_coords, y_coords, _ = get_spatial_coords(adata_gene, sample)
            if 0 <= root_idx < len(x_coords):
                root_coord = (float(x_coords[root_idx]), float(y_coords[root_idx]))
                adata_gene.uns['iroot'] = root_idx
                adata_img.uns['iroot'] = root_idx
            else:
                raise ValueError(f"Invalid root node index:{root_idx}")
     
            # ========== Step 7: Check the connectivity of the genetic data graph ==========
            if 'connectivities' in adata_gene.obsp:
                G_gene = adata_gene.obsp['connectivities']
            else:
                metrics["reason"] = "Missing gene connectivities"
                summary_rows.append([metrics[col] for col in summary_columns])
                continue
            
            n_comp_gene, labels_gene = connected_components(G_gene, directed=False, connection='weak')
            if n_comp_gene != 1:
                metrics["reason"] = f"Gene graph not connected ({n_comp_gene} components)"
                summary_rows.append([metrics[col] for col in summary_columns])
                continue

            # ========== Step 8: Check the connectivity of the image data graph ==========
            if 'connectivities' in adata_img.obsp:
                G_img = adata_img.obsp['connectivities']
                n_comp_img, _ = connected_components(G_img, directed=False, connection='weak')
                if n_comp_img != 1:
                    metrics["reason"] = f"Image graph not connected ({n_comp_img} components)"
                    summary_rows.append([metrics[col] for col in summary_columns])
                    continue
            else:
                metrics["reason"] = "Missing image connectivities"
                summary_rows.append([metrics[col] for col in summary_columns])
                continue
            if seg_available:
                tumor_ratio_vals = adata_gene.obs['tumor_ratio'].values.astype(float)
                metrics["tumor_ratio_mean"] = np.mean(tumor_ratio_vals)
                metrics["tumor_ratio_max"] = np.max(tumor_ratio_vals)
                metrics["tumor_ratio_median"] = np.median(tumor_ratio_vals)
     
            # ========== Step 9: Calculate gene pseudotime ==========
            try:
                sc.tl.dpt(adata_gene, n_dcs=10)
            except TypeError:
                sc.tl.dpt(adata_gene)
            gene_ptime = adata_gene.obs['dpt_pseudotime'].values.copy()
 
            gene_ptime_valid, gene_nan_count, gene_inf_count = check_pseudotime_validity_strict(gene_ptime, sample, "基因")
            if not gene_ptime_valid:
                metrics["reason"] = "Gene pseudotime invalid"
                summary_rows.append([metrics[col] for col in summary_columns])
                continue
            sc.tl.umap(adata_gene, min_dist=0.5, spread=1.0)
            gene_ptime_clean = gene_ptime[np.isfinite(gene_ptime)]
            metrics["gene_ptime_mean"] = np.mean(gene_ptime_clean)
            metrics["gene_ptime_std"] = np.std(gene_ptime_clean)
            metrics["gene_ptime_min"] = np.min(gene_ptime_clean)
            metrics["gene_ptime_max"] = np.max(gene_ptime_clean)
            
            # ========== Step 10: Calculate image pseudotime ==========
            img_pseudotime_success = True
            img_ptime = None
            try:
                sc.tl.dpt(adata_img, n_dcs=10)
                img_ptime = adata_img.obs['dpt_pseudotime'].values.copy()
                adata_img.obs['image_pseudotime'] = img_ptime
                
                img_ptime_valid, img_nan_count, img_inf_count = check_pseudotime_validity_strict(img_ptime, sample, "图像")
                if not img_ptime_valid:
                    img_pseudotime_success = False
                    img_ptime = None
                else:
                    sc.tl.umap(adata_img, min_dist=0.5, spread=1.0)
        
                    img_ptime_clean = img_ptime[np.isfinite(img_ptime)]
                    metrics["img_ptime_mean"] = np.mean(img_ptime_clean)
                    metrics["img_ptime_std"] = np.std(img_ptime_clean)
                    metrics["img_ptime_min"] = np.min(img_ptime_clean)
                    metrics["img_ptime_max"] = np.max(img_ptime_clean)
            except Exception as e:
                img_pseudotime_success = False
                img_ptime = None

            # ========== Step 11: Collect and store the results ==========
            save_umap_results(
                adata_gene=adata_gene,
                adata_img=adata_img,
                gene_ptime=gene_ptime,
                img_ptime=img_ptime,
                sample_name=sample,
                save_dir=save_dir
            )
            save_patch_combined_results(
                adata_gene=adata_gene,
                adata_img=adata_img,
                pca_result_gene=adata_gene.obsm['X_pca'],
                pca_result_img=adata_img.obsm['X_pca'] if 'X_pca' in adata_img.obsm else None,
                gene_ptime=gene_ptime,
                img_ptime=img_ptime,
                cell_composition_df=cell_composition_df,
                sample_name=sample,
                save_dir=save_dir,
                root_idx=root_idx 
            )

            plot_combined_pca_with_celltypes_separate(
                adata=adata_img,
                pca_result=adata_img.obsm['X_pca'],
                pseudotime=img_ptime,
                cell_composition_df=cell_composition_df,
                sample_name=sample,
                save_dir=save_dir,
                modality="image",
                figsize=(8, 7),
                s=30,
                cell_types=['connective', 'neoplastic']
            )            

            compute_celltype_pseudotime_correlations_full(
                adata_gene=adata_gene,
                adata_img=adata_img,
                cell_composition_df=cell_composition_df,
                gene_ptime=gene_ptime,
                img_ptime=img_ptime,
                sample_name=sample,
                save_dir=save_dir
            )

            if img_pseudotime_success and img_ptime is not None:
               
                corr_metrics = compute_correlation_metrics(gene_ptime, img_ptime)
                ranking_metrics = compute_ranking_metrics(gene_ptime, img_ptime)
                dist_metrics = compute_distribution_metrics(gene_ptime, img_ptime)
                cos_sim = compute_cosine_similarity(gene_ptime, img_ptime)
                
                metrics.update(corr_metrics)
                metrics.update(ranking_metrics)
                metrics.update(dist_metrics)
                metrics["cosine_similarity"] = cos_sim

                plot_image_pseudotime_distribution(img_ptime, sample, save_dir)
                adata_img.obs['image_pseudotime'] = img_ptime
                save_spatial_with_root(
                    adata_for_plot=adata_img,
                    color_key='image_pseudotime',
                    root_coord=root_coord,
                    save_path=os.path.join(save_dir, f"{sample}_image_pseudotime_spatial.png"),
                    sample_name=sample
                )
                img_ptime_df = pd.DataFrame({
                    "barcode": adata_img.obs_names,
                    "image_pseudotime": img_ptime,
                    "tumor_ratio": adata_img.obs.get('tumor_ratio', np.nan),
                    "cell_count": adata_img.obs.get('cell_count', np.nan),
                    "leiden_cluster": adata_img.obs.get('leiden', np.nan),
                    "spatial_x": x_coords,
                    "spatial_y": y_coords,
                })
                img_ptime_path = os.path.join(save_dir, f"{sample}_image_pseudotime_results.csv")
                img_ptime_df.to_csv(img_ptime_path, index=False, encoding="utf-8-sig")
                
                if cell_composition_df is not None and not cell_composition_df.empty:
                    spot_cell_counts = cell_composition_df.groupby(['spot_barcode', 'cell_type']).size().reset_index(name='count')
                    spot_total_counts = spot_cell_counts.groupby('spot_barcode')['count'].sum().reset_index(name='total_count')
                    spot_cell_ratios = spot_cell_counts.merge(spot_total_counts, on='spot_barcode')
                    spot_cell_ratios['ratio'] = spot_cell_ratios['count'] / spot_cell_ratios['total_count']
                    
                    cell_type_ratio_pivot = spot_cell_ratios.pivot_table(
                        index='spot_barcode', 
                        columns='cell_type', 
                        values='ratio',
                        fill_value=0
                    ).reset_index()
                    
                    barcodes = adata_gene.obs_names.astype(str).tolist()
                    cell_type_ratio_pivot = cell_type_ratio_pivot.set_index('spot_barcode').reindex(barcodes).fillna(0).reset_index()
                    
                    valid_cell_types = [col for col in cell_type_ratio_pivot.columns 
                                       if col != 'spot_barcode' and col.lower() != 'unknown']
                    valid_cell_types = [ct for ct in valid_cell_types if cell_type_ratio_pivot[ct].sum() > 0]
                    
                    # ---------- HVG PCA Image ----------
                    if GENE_SELECTION_MODE == "SVG":
                        adata_hvg_for_pca = adata_gene_log1p.copy()
                        sc.pp.highly_variable_genes(adata_hvg_for_pca, n_top_genes=HVG_N)
                        n_hvg_check = int(adata_hvg_for_pca.var['highly_variable'].sum())
                        
                        if n_hvg_check >= N_PCS:
                            adata_hvg_for_pca = adata_hvg_for_pca[:, adata_hvg_for_pca.var['highly_variable']].copy()
                            if hasattr(adata_hvg_for_pca.X, 'toarray'):
                                X_hvg = adata_hvg_for_pca.X.toarray()
                            else:
                                X_hvg = np.array(adata_hvg_for_pca.X)
                            
                            from sklearn.decomposition import PCA
                            n_comps_hvg = min(4, X_hvg.shape[1], X_hvg.shape[0] - 1)
                            if n_comps_hvg >= 2:
                                pca_hvg = PCA(n_components=n_comps_hvg)
                                pca_result_hvg = pca_hvg.fit_transform(X_hvg)
                                
                                adata_hvg_plot = sc.AnnData(X_hvg)
                                adata_hvg_plot.obs = adata_gene.obs.copy()
                                adata_hvg_plot.obsm['X_pca'] = pca_result_hvg
                                adata_hvg_plot.uns['pca'] = {'variance_ratio': pca_hvg.explained_variance_ratio_}
                                
                                plot_pca_pseudotime_with_cell_types(
                                    adata=adata_hvg_plot,
                                    pca_result=pca_result_hvg,
                                    pseudotime=gene_ptime,
                                    cell_composition_df=cell_composition_df,
                                    sample_name=sample,
                                    save_dir=save_dir,
                                    modality="gene",
                                    gene_set="HVG",
                                    figsize=None,
                                    s=20,
                                    root_idx=root_idx
                                )
                                
                                corr_hvg, var_hvg = compute_pc_celltype_correlations(
                                    pca_result=pca_result_hvg,
                                    cell_type_ratio_pivot=cell_type_ratio_pivot,
                                    valid_cell_types=valid_cell_types,
                                    sample_name=sample,
                                    save_dir=save_dir,
                                    modality="gene",
                                    gene_set="HVG",
                                    pca_variance_ratio=pca_hvg.explained_variance_ratio_
                                )
                            else:
                                corr_hvg = None
                                var_hvg = None
                        else:
                            corr_hvg = None
                            var_hvg = None
                    else:
                        var_hvg = pd.DataFrame({
                            'PC': [f'PC{i+1}' for i in range(4)],
                            'explained_variance_ratio': adata_gene.uns['pca']['variance_ratio'][:4],
                            'explained_variance_percent': adata_gene.uns['pca']['variance_ratio'][:4] * 100
                        })
                        
                        plot_pca_pseudotime_with_cell_types(
                            adata=adata_gene,
                            pca_result=adata_gene.obsm['X_pca'],
                            pseudotime=gene_ptime,
                            cell_composition_df=cell_composition_df,
                            sample_name=sample,
                            save_dir=save_dir,
                            modality="gene",
                            gene_set="HVG",
                            figsize=None,
                            s=20,
                            root_idx=root_idx
                        )
                        
                        corr_hvg, _ = compute_pc_celltype_correlations(
                            pca_result=adata_gene.obsm['X_pca'],
                            cell_type_ratio_pivot=cell_type_ratio_pivot,
                            valid_cell_types=valid_cell_types,
                            sample_name=sample,
                            save_dir=save_dir,
                            modality="gene",
                            gene_set="HVG",
                            pca_variance_ratio=adata_gene.uns['pca']['variance_ratio'][:4]
                        )
                    
                    # ---------- SVG PCA Image ----------
                    if len(svg_genes) >= 10:
                        available_svg = [g for g in svg_genes if g in adata_gene_log1p.var_names]
                        if len(available_svg) >= 10:
                            adata_svg_for_pca = adata_gene_log1p[:, available_svg].copy()   
                            if hasattr(adata_svg_for_pca.X, 'toarray'):
                                X_svg = adata_svg_for_pca.X.toarray()
                            else:
                                X_svg = np.array(adata_svg_for_pca.X)

                            n_comps_svg = min(4, X_svg.shape[1], X_svg.shape[0] - 1)
                            if n_comps_svg >= 2:
                                pca_svg = PCA(n_components=n_comps_svg)
                                pca_result_svg = pca_svg.fit_transform(X_svg)
                                
                                adata_svg_plot = sc.AnnData(X_svg)
                                adata_svg_plot.obs = adata_gene.obs.copy()
                                adata_svg_plot.obsm['X_pca'] = pca_result_svg
                                adata_svg_plot.uns['pca'] = {'variance_ratio': pca_svg.explained_variance_ratio_}
                                
                                plot_pca_pseudotime_with_cell_types(
                                    adata=adata_svg_plot,
                                    pca_result=pca_result_svg,
                                    pseudotime=gene_ptime,
                                    cell_composition_df=cell_composition_df,
                                    sample_name=sample,
                                    save_dir=save_dir,
                                    modality="gene",
                                    gene_set="SVG",
                                    figsize=None,
                                    s=20,
                                    root_idx=root_idx
                                )
                                
                                var_svg = pd.DataFrame({
                                    'PC': [f'PC{i+1}' for i in range(n_comps_svg)],
                                    'explained_variance_ratio': pca_svg.explained_variance_ratio_,
                                    'explained_variance_percent': pca_svg.explained_variance_ratio_ * 100
                                })
                                
                                corr_svg, _ = compute_pc_celltype_correlations(
                                    pca_result=pca_result_svg,
                                    cell_type_ratio_pivot=cell_type_ratio_pivot,
                                    valid_cell_types=valid_cell_types,
                                    sample_name=sample,
                                    save_dir=save_dir,
                                    modality="gene",
                                    gene_set="SVG",
                                    pca_variance_ratio=pca_svg.explained_variance_ratio_
                                )
                            else:
                                corr_svg = None
                                var_svg = None
                        else:
                            corr_svg = None
                            var_svg = None
                    else:
                        corr_svg = None
                        var_svg = None
                    # ----------  Cell type - gene pseudo-time correlation ----------
                    corr_ptime_gene = compute_celltype_pseudotime_correlations(
                        pseudotime=gene_ptime,
                        cell_type_ratio_pivot=cell_type_ratio_pivot,
                        valid_cell_types=valid_cell_types,
                        sample_name=sample,
                        save_dir=save_dir,
                        modality="gene"
                    )
                    plot_celltype_pca_combined_hvg(
                        adata=adata_gene,
                        pca_result=adata_gene.obsm['X_pca'],
                        pseudotime=gene_ptime,
                        cell_composition_df=cell_composition_df,
                        sample_name=sample,
                        save_dir=save_dir,
                        modality="gene",
                        gene_set="HVG",
                        figsize=(18, 6),
                        s=20,
                        root_idx=root_idx
                    )

                    plot_celltype_pca_combined(
                        adata=adata_img,
                        pca_result=adata_img.obsm['X_pca'],
                        pseudotime=img_ptime,
                        cell_composition_df=cell_composition_df,
                        sample_name=sample,
                        save_dir=save_dir,
                        modality="image",
                        gene_set="",
                        figsize=(18, 6),
                        s=20,
                        root_idx=root_idx
                    )
                    # ---------- Image PCA ----------
                    var_img = pd.DataFrame({
                        'PC': [f'PC{i+1}' for i in range(4)],
                        'explained_variance_ratio': adata_img.uns['pca']['variance_ratio'][:4],
                        'explained_variance_percent': adata_img.uns['pca']['variance_ratio'][:4] * 100
                    })
                    plot_pca_pseudotime_with_cell_types(
                        adata=adata_img,
                        pca_result=adata_img.obsm['X_pca'],
                        pseudotime=img_ptime,
                        cell_composition_df=cell_composition_df,
                        sample_name=sample,
                        save_dir=save_dir,
                        modality="image",
                        gene_set="",
                        figsize=None,
                        s=20,
                        root_idx=root_idx
                    )
                    corr_img, _ = compute_pc_celltype_correlations(
                        pca_result=adata_img.obsm['X_pca'],
                        cell_type_ratio_pivot=cell_type_ratio_pivot,
                        valid_cell_types=valid_cell_types,
                        sample_name=sample,
                        save_dir=save_dir,
                        modality="image",
                        gene_set="",
                        pca_variance_ratio=adata_img.uns['pca']['variance_ratio'][:4]
                    )
                    
                    corr_ptime_img = compute_celltype_pseudotime_correlations(
                        pseudotime=img_ptime,
                        cell_type_ratio_pivot=cell_type_ratio_pivot,
                        valid_cell_types=valid_cell_types,
                        sample_name=sample,
                        save_dir=save_dir,
                        modality="image"
                    )

                    save_all_correlation_summary(
                        corr_hvg=corr_hvg,
                        corr_svg=corr_svg,
                        corr_img=corr_img,
                        corr_ptime_gene=corr_ptime_gene,
                        corr_ptime_img=corr_ptime_img,
                        var_hvg=var_hvg,
                        var_svg=var_svg,
                        var_img=var_img,
                        sample_name=sample,
                        save_dir=save_dir
                    )
                adata_img_list.append(adata_img)
                sample_names_list.append(sample)

            metrics["n_valid_spots"] = adata_gene.n_obs
            
            plot_gene_pseudotime_distribution(gene_ptime, sample, save_dir)
            adata_gene.obs['gene_pseudotime'] = gene_ptime
            save_spatial_with_root(
                adata_for_plot=adata_gene,
                color_key='gene_pseudotime',
                root_coord=root_coord,
                save_path=os.path.join(save_dir, f"{sample}_gene_pseudotime_spatial.png"),
                sample_name=sample
            )
            plot_tumor_ratio_spatial(
                adata_before=adata_before_cell_filter,
                adata_after=adata_gene,
                sample_name=sample,
                save_dir=save_dir,
                sample_dir=sample_dir
            )

            gene_ptime_df = pd.DataFrame({
                "barcode": adata_gene.obs_names,
                "gene_pseudotime": gene_ptime,
                "tumor_ratio": adata_gene.obs.get('tumor_ratio', np.nan),
                "cell_count": adata_gene.obs.get('cell_count', np.nan),
                "leiden_cluster": adata_gene.obs.get('leiden', np.nan),
                "spatial_x": x_coords,
                "spatial_y": y_coords,
            })
            gene_ptime_path = os.path.join(save_dir, f"{sample}_gene_pseudotime_results.csv")
            gene_ptime_df.to_csv(gene_ptime_path, index=False, encoding="utf-8-sig")

            df_res = adata_gene.obs[['dpt_pseudotime', 'tumor_ratio', 'tumor_count', 'cell_count', 'leiden']].copy()
            df_res.rename(columns={'dpt_pseudotime': 'gene_pseudotime'}, inplace=True)
            df_res['image_pseudotime'] = img_ptime if img_ptime is not None else np.nan
            df_res['spot_barcode'] = adata_gene.obs_names.values
            df_res['spatial_x'] = x_coords
            df_res['spatial_y'] = y_coords
            
            df_res['gene_set_used'] = gene_set_used
            df_res['n_genes_used'] = n_genes_used
            
            for col in [
                'pearson_r', 'pearson_p', 'kendall_tau', 'kendall_p', 'spearman_r', 'spearman_p',
                'distance_corr', 'kendall_tau_b', 'kendall_tau_b_p', 'tau_distance',
                'concordant_pairs', 'discordant_pairs', 'wasserstein_distance', 'cosine_similarity',
                'tumor_ratio_mean', 'tumor_ratio_max', 'tumor_ratio_median',
                'cell_count_mean', 'cell_count_max', 'cell_count_median',
                'zero_cell_spots', 'zero_cell_ratio',
                'total_cells_matched', 'unique_cell_types', 'avg_cells_per_spot',
                'final_cell_threshold', 'leiden_clusters', 'start_cluster_id',
                'start_cluster_tumor_ratio', 'start_cluster_size',
                'original_spots', 'after_qc', 'after_tissue_filter', 'after_cell_filter', 'after_all_filters',
                'retention_rate_qc', 'retention_rate_tissue', 'retention_rate_cell', 'retention_rate_total'
            ]:
                df_res[col] = metrics[col]
            
            df_res.to_csv(os.path.join(save_dir, f"{sample}_detailed_results.csv"), index=False, encoding="utf-8-sig")

            metrics_summary = pd.DataFrame({col: [metrics[col]] for col in summary_columns if col != "sample"})
            metrics_summary['gene_set_used'] = gene_set_used
            metrics_summary['n_genes_used'] = n_genes_used
            metrics_summary.to_csv(os.path.join(save_dir, f"{sample}_metrics_summary.csv"), index=False, encoding="utf-8-sig")
            summary_rows.append([metrics[col] for col in summary_columns])
            
        except Exception as e:
            metrics["reason"] = f"Processing error: {str(e)[:100]}"
            print(f"Processing of the sample failed:{e}")
            import traceback
            traceback.print_exc()
            summary_rows.append([metrics[col] for col in summary_columns])
    
    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    summary_path = os.path.join(SAVE_ROOT, "summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nAll sample processing is completed and the summary table has been saved at:{summary_path}")