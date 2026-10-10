<!-- Title -->
<div align="center">
<h1>Benchmarking pathological foundation models morphologically and molecularly using pseudotime trajectories</h1>

</div>

Note that you need to download all required datasets and pretrained models by yourself using the links provided at the end, and update the corresponding paths before running the scripts.

## How to install
Follow the steps below to create and activate the environment, and install the project dependencies:
```python
conda create --name trajspace python=3.10
conda activate trajspace
pip install -r requirements.txt
```

## Patch extraction
You can use the following commands to extract image patches from the HEST dataset, either at the spot size or at a fixed resolution:
```python
python preprocessing/patch_extractor.py
```

## Feature extraction
For standard architectures such as ResNet18 and ViT-B/16, we leverage the [timm](https://huggingface.co/timm) library. For other specialized models, we use the [Trident](https://github.com/mahmoodlab/TRIDENT) framework. You can use the following commands to extract features:

```python
python preprocessing/feature_extractor.py
```

## Filtered data

Run the code to process the filtered sample data:

```
python pesudotime/dataset_process.py
```

## Pseudotime calculation

The sample is based on ST data and pseudotime calculation based on image features:

```
python pesudotime/Cal_PT.py
```


## Pseudotime-based gene enrichment

Enrichment of genes based on pseudo-time trajectory:

```
python pesudotime/Gene_Enrich.py
```


## Visualized genes
The spatial distribution of visualized genes:

```
python pesudotime/Gene_Expression.py
```


## Pathology to spatial transcriptomics
For the pathology-to-spatial transcriptomics prediction task, we adopt the method proposed in the [MISO](https://www.nature.com/articles/s41467-025-66691-y) paper.
The official implementation is available at [here](https://github.com/owkin/miso_code).


## HEST data download link

| Dataset                  |  Publication | Download Link |
|--------------------------|--------------|--------------|
| HEST-1k                  | NIPS-26  | [Link](https://github.com/mahmoodlab/HEST) |


## Pretrained model download link
Note: Access to each model must be requested before downloading.

| Methods                  |  Publication | Download Link |
|--------------------------|--------------|--------------|
| Resnet18                 | CVPR 16 |[Link](https://huggingface.co/timm) |
| ViT-B/16                 | ICLR 21 |[Link](https://huggingface.co/timm) |
| CTransPath               | MedIA 22 |[Link](https://github.com/Xiyue-Wang/TransPath) |
| CONCH                    | Nat. Med. 24 |[Link](https://huggingface.co/MahmoodLab/CONCH/tree/main) |
| CONCHv1.5                | - |[Link](https://huggingface.co/MahmoodLab/conchv1_5) |
| Prov-GiagPath            | Nature 24 |[Link](https://huggingface.co/prov-gigapath/prov-gigapath) |
| UNI                      | Nat. Med. 24 |[Link](https://huggingface.co/MahmoodLab/UNI) |
| UNI2-h                   | - |[Link](https://huggingface.co/MahmoodLab/UNI2-h) |
| Virchow                  | Nat. Med. 24 |[Link](https://huggingface.co/paige-ai/Virchow) |
| Virchow2                 | - |[Link](https://huggingface.co/paige-ai/Virchow2) |

