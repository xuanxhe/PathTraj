import os
import shutil
import zipfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm

#Downloaded dataset directory
RAW_ROOT       = r"hest_data"
#samples directory
CSV_PATH       = r"select_samples.csv"
#Filtered sample list
OUTPUT_ROOT    = r"samples"
# ====================================================

def get_sample_list():
    df = pd.read_csv(CSV_PATH)
    samples = [str(s).strip() for s in df.iloc[:, 0].dropna()]
    return sorted(set(samples))

def process():
    sample_list = get_sample_list()
    raw_path = Path(RAW_ROOT)
    top_folders = [f for f in raw_path.iterdir() if f.is_dir() and not f.name.startswith(".")]

    print(f"samples:{len(sample_list)}")
    print(f"Function Folder:{[f.name for f in top_folders]}")

    for sample in tqdm(sample_list, desc="Process the sample"):
        sample_out = Path(OUTPUT_ROOT) / sample
        sample_out.mkdir(parents=True, exist_ok=True)

        for top_folder in top_folders:
            top_name = top_folder.name
            all_files = list(top_folder.rglob("*"))

            for f in all_files:
                if not f.is_file():
                    continue

                filename = f.name
                if sample not in filename:
                    continue

                target_dir = sample_out / top_name
                target_dir.mkdir(parents=True, exist_ok=True)
                suffix = f.suffix
                parent_dir = f.parent.name  

                # ===================== Renaming rules =====================
                if top_name == "ext_feature":
                    if "patches_fix" in str(f):
                        new_name = f"{sample}_patches_fix{suffix}"
                    elif "patches_spot" in str(f):
                        new_name = f"{sample}_patches_spot{suffix}"
                    else:
                        new_name = filename

                elif top_name == "ext_patch":
                    new_name = f"{sample}_{parent_dir}{suffix}"


                elif top_name == "patches":
                    new_name = f"{sample}{suffix}"

                else:
                    new_name = filename
                # ========================================================

                dest = target_dir / new_name
                if not dest.exists():
                    shutil.copy2(f, dest)

                # cellvit_seg decompresses and deletes the zip file
                if top_name == "cellvit_seg" and dest.suffix == ".zip":
                    try:
                        with zipfile.ZipFile(dest, 'r') as zf:
                            zf.extractall(target_dir)
                        dest.unlink()
                    except:
                        pass

    print("\nAll processing is complete")
    print("Output path:", OUTPUT_ROOT)

if __name__ == "__main__":
    process()
