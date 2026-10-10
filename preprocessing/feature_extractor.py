import gc
import glob
import timm
import h5py
import torch
import argparse
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from trident.patch_encoder_models import encoder_factory


def parse_args():
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument("--name", type=str, default="all", help="None")
    parser.add_argument("--ckpt-path", type=str, default="/home/andrewtal/SSD1/projects/PHST/ckpts", help="None")
    parser.add_argument("--data-path", type=str, default="/home/andrewtal/SSD1/projects/PHST/patch_ee/data/HEST-1k/patches_ext", help="None")
    return parser.parse_args()


class ImagePathDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform

    def __len__(self):
        with h5py.File(self.h5_path, 'r') as f:
            length = len(f['img'])
        return length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            img = Image.fromarray(f['img'][idx])
        if self.transform:
            img = self.transform(img)
        
        return img



def get_model(model_name, ckpt_path):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    if model_name == 'RN18':
        model = timm.create_model(
            'resnet18', 
            pretrained=False, 
            checkpoint_path='{}/resnet18/models--timm--resnet18.a1_in1k/snapshots/491b427b45c94c7fb0e78b5474cc919aff584bbf/model.safetensors'.format(ckpt_path)
        )
        model.fc = nn.Identity()
    elif model_name == 'ViTB16':
        model = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=False,
            checkpoint_path='../ckpts/vitb16/models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/snapshots/063c6c38a5d8510b2e57df480445e94b231dad2c/model.safetensors'
        )
        model.head = nn.Identity()
    elif model_name == 'CONCH15':
        model = encoder_factory('conch_v15', weights_path='{}/conch15/pytorch_model_vision.bin'.format(ckpt_path))
    elif model_name == 'CONCH':
        model = encoder_factory('conch_v1', weights_path='{}/conch/pytorch_model.bin'.format(ckpt_path))
    elif model_name == 'GigaPath':
        model = encoder_factory('gigapath', weights_path='{}/provgigapath/pytorch_model.bin'.format(ckpt_path))
    elif model_name == 'Virchow':
        model = encoder_factory('virchow', weights_path='{}/virchow/pytorch_model.bin'.format(ckpt_path))
    elif model_name == 'Virchow2':
        model = encoder_factory('virchow2', weights_path='{}/virchow2/pytorch_model.bin'.format(ckpt_path))
    elif model_name == 'CTransPath':
        model = encoder_factory('ctranspath', weights_path='{}/CTransPath/CHIEF_CTransPath.pth'.format(ckpt_path))
    elif model_name == 'UNI':
        model = encoder_factory('uni_v1', weights_path='{}/uni/models--MahmoodLab--uni/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin'.format(ckpt_path))
    elif model_name == 'UNI2h':
        model = encoder_factory('uni_v2', weights_path='{}/uni2h/pytorch_model.bin'.format(ckpt_path))
    else:
        print('Error!')
    
    model.eval().cuda()
    return model, transform


def predict_one_h5(h5_path, model, transform, model_name):
    with h5py.File(h5_path, 'a') as f:
        if '{}_features'.format(model_name) in f:
            print("{}_features exist!".format(model_name))
            return
    
    dataset = ImagePathDataset(h5_path, transform)
    loader = DataLoader(dataset, batch_size=32, num_workers=8, pin_memory=True, shuffle=False)
    
    all_feats = []
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.cuda()
            feats = model(imgs).cpu()
            all_feats.append(feats)
    all_feats = torch.cat(all_feats, dim=0).numpy()

    with h5py.File(h5_path, 'a') as f:
        f.create_dataset('{}_features'.format(model_name), data=all_feats)
        print("{}_features appended！".format(model_name))



if __name__ == "__main__":
    args = parse_args()
    h5_paths = glob.glob('{}/**/*.h5'.format(args.data_path), recursive=True)
    print('File Number:', len(h5_paths))
    
    for model_name in ['RN18', 'ViTB16', 'CTransPath', 'CONCH', 'UNI', 'GigaPath', 'CONCH15', 'UNI2h', 'Virchow', 'Virchow2']:
        model, transform = get_model(model_name, args.ckpt_path)
        for path in tqdm(h5_paths, total=len(h5_paths)):
            print(model_name, path)
            predict_one_h5(path, model, transform, model_name)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
